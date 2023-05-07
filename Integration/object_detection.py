# ==============================================================================
# -- ModelPrediction -----------------------------------------------------------
# ==============================================================================

import random

import cv2
import numpy as np
import torch

import tools
from commons import CONSTANTS
from yolov7.utils.datasets import letterbox
from yolov7.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov7.utils.plots import plot_one_box

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

focus_labels = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'train', 'truck',
                'traffic light', 'stop sign']

######################################################################################
# camera parameters

# Generate a grid of pixel coordinates
u = np.indices((int(CONSTANTS.WINDOW_HEIGHT), int(CONSTANTS.WINDOW_WIDTH)))[1]
v = np.indices((int(CONSTANTS.WINDOW_HEIGHT), int(CONSTANTS.WINDOW_WIDTH)))[0]

Center_X = int(CONSTANTS.WINDOW_WIDTH / 2)
Center_Y = int(CONSTANTS.WINDOW_HEIGHT / 2)

fov = 90.0
focal = CONSTANTS.WINDOW_WIDTH / (2.0 * np.tan(fov * np.pi / 360.0))


######################################################################################

def generate_3D_map(_depth_map):
    # Compute 3D x and y coordinates
    x = (u - Center_X) * _depth_map / focal
    y = (v - Center_Y) * _depth_map / focal
    z = _depth_map

    return x, y, z


def detect(model, input_image, _depth_map, imgsz):
    xyz_3D = generate_3D_map(_depth_map)

    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = random.choices(tools.get_colors(), k=len(names))

    ##########################################################

    # Padded resize
    img = letterbox(input_image, imgsz, stride=stride)[0]
    # Convert
    img = img.transpose(2, 0, 1)  # 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0

    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
        pred = model(img, augment=False)[0]

    # Apply NMS
    pred = non_max_suppression(pred, 0.55, 0.55)

    x_3D, y_3D, z_3D = xyz_3D

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], input_image.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                if names[int(cls)] in focus_labels:
                    # 2d box coordinates
                    x_min = int(xyxy[0])
                    y_min = int(xyxy[1])
                    x_max = int(xyxy[2])
                    y_max = int(xyxy[3])
                    c1, c2 = (x_min, y_min), (x_max, y_max)

                    # 3d box coordinates
                    x_box = x_3D[y_min:y_max, x_min:x_max].reshape(-1)
                    y_box = y_3D[y_min:y_max, x_min:x_max].reshape(-1)
                    z_box = z_3D[y_min:y_max, x_min:x_max].reshape(-1)

                    distance = np.sqrt(x_box ** 2 + y_box ** 2 + z_box ** 2)
                    min_distance_idx = np.argmin(distance)
                    min_distance = distance[min_distance_idx]

                    label = f'{names[int(cls)]} {conf:.2f} {min_distance:.2f}' + 'm'

                    plot_one_box(c1, c2, input_image, label=label, color=colors[int(cls)], line_thickness=1)

                    if conf > 0.9:
                        if names[int(cls)] == 'car':
                            if __car_collision(x_box, min_distance, threshold=10):
                                print("Slow down!A Car Ahead of You.")

                        elif names[int(cls)] == 'stop sign':
                            print("Stop Sign Detected!")
                        elif names[int(cls)] == 'traffic light':
                            if min_distance < 20:
                                traffic_light = __get_roi(input_image, c1, c2)
                                traffic_light_color = __get_red_or_green(traffic_light)
                                print(f"Traffic light {traffic_light_color} color")
    del img


def __car_collision(x_box, distance, threshold):
    return np.abs(np.min(x_box)) < 5 and distance < threshold


def __get_roi(image, c1, c2):
    x1, y1 = c1
    x2, y2 = c2
    roi = image[y1:y2, x1:x2]
    return roi


def __get_red_or_green(traffic_light):
    count_red = __count_red(traffic_light)
    count_green = __count_green(traffic_light)
    num_pixels = np.size(traffic_light) // 3

    if count_green > count_red and count_green > num_pixels // 20:
        return "GREEN"
    elif count_red > count_green and count_red > num_pixels // 20:
        return "RED"
    else:
        return "UNSURE"


def __count_red(img):
    # Define the red color range in HSV color space
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    lower_red_2 = np.array([170, 50, 50])
    upper_red_2 = np.array([180, 255, 255])

    # Convert the input image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Create a mask for pixels that fall within the red color range
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    mask = mask1 | mask2
    return np.count_nonzero(mask)


def __count_green(img):
    # Define the green color range in HSV color space
    lower_green = np.array([36, 25, 25])
    upper_green = np.array([70, 255, 255])

    # Convert the input image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Create a mask for pixels that fall within the green color range
    mask = cv2.inRange(hsv, lower_green, upper_green)

    return np.count_nonzero(mask)
