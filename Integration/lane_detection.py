import cv2
import numpy as np


# colors_platte = np.array([[0, 0, 0],  # Unlabeled
#                           [70, 70, 70],  # Building
#                           [100, 40, 40],  # Fence
#                           [55, 90, 80],  # Other -> Everything that does not belong to any other category.
#                           [220, 20, 60],  # Pedestrian
#                           [153, 153, 153],  # Pole
#                           [157, 234, 50],  # Roadline
#                           [128, 64, 128],  # Road
#                           [244, 35, 232],  # Sidewalk
#                           [107, 142, 35],  # Vegetation
#                           [0, 0, 142],  # Car
#                           [102, 102, 156],  # Wall
#                           [220, 220, 0],  # Traffic sign
#                           # not used in current model
#                           [70, 130, 180],  # Sky
#                           [81, 0, 81],  # Ground
#                           [150, 100, 100],  # Bridge
#                           [230, 150, 140],  # RailTrack
#                           [180, 165, 180],  # GuardRail
#                           [250, 170, 30],  # TrafficLight
#                           [110, 190, 160],  # Static
#                           [170, 120, 50],  # Dynamic
#                           [45, 60, 150],  # Water
#                           [145, 170, 100]]  # Terrain
#                          )


def estimate_lane_lines(_main_image, _segmentation_mask):
    # cv2.imwrite("/home/oraby/Pictures/presentation/lane_detection/1_main_image.png", _main_image[:, :, ::-1])
    # cv2.imwrite("/home/oraby/Pictures/presentation/lane_detection/2_seg_mask.png",
    #             colors_platte[_segmentation_mask][:, :, ::-1])

    # Create an image with pixels belonging to lane boundary categories from the output of semantic segmentation
    lane_mask = np.zeros(_segmentation_mask.shape, dtype=np.uint8)
    lane_mask[_segmentation_mask == 8] = 255
    lane_mask[_segmentation_mask == 6] = 255

    # cv2.imwrite("/home/oraby/Pictures/presentation/lane_detection/3_lane_mask.png", lane_mask)

    # Perform Edge Detection
    edges = cv2.Canny(lane_mask, 50, 150, apertureSize=3)

    # cv2.imwrite("/home/oraby/Pictures/presentation/lane_detection/4_edges.png", edges)

    img_shape = _main_image.shape
    vertices = np.array(
        [[(0, img_shape[0]), (img_shape[1] * 0.45, img_shape[0] * 0.4), (img_shape[1] * 0.55, img_shape[0] * 0.4),
          (img_shape[1], img_shape[0])]], dtype=np.int32)
    roi = region_of_interest(edges, vertices)

    # cv2.imwrite("/home/oraby/Pictures/presentation/lane_detection/5_roi.png", roi)

    # Perform Line estimation
    lines = cv2.HoughLinesP(roi, rho=1, theta=np.pi / 180, threshold=100, minLineLength=20, maxLineGap=400)

    # lane_lines_image = np.zeros(_main_image.shape, dtype=np.uint8)
    # vis_lanes(lane_lines_image, lines, thickness=1)
    # cv2.imwrite("/home/oraby/Pictures/presentation/lane_detection/6_lane_lines.png",
    #             cv2.cvtColor(lane_lines_image, cv2.COLOR_RGB2GRAY))

    left_lane_lines, right_lane_lines = filter_lines(lines)

    # filtered_lane_lines_image = np.zeros(_main_image.shape, dtype=np.uint8)
    # vis_lanes(filtered_lane_lines_image, left_lane_lines, thickness=1)
    # vis_lanes(filtered_lane_lines_image, right_lane_lines, thickness=1)
    # cv2.imwrite("/home/oraby/Pictures/presentation/lane_detection/7_filtered_lane_lines.png",
    #             cv2.cvtColor(filtered_lane_lines_image, cv2.COLOR_RGB2GRAY))

    merged_lines = merge_lines(left_lane_lines, right_lane_lines)

    # merged_lane_lines_image = np.zeros(_main_image.shape, dtype=np.uint8)
    # vis_lanes(merged_lane_lines_image, filtered_lines, thickness=5)
    # cv2.imwrite("/home/oraby/Pictures/presentation/lane_detection/8_merged_lane_lines.png",
    #             cv2.cvtColor(merged_lane_lines_image, cv2.COLOR_RGB2GRAY))

    vis_lanes(_main_image, merged_lines, thickness=5)
    # cv2.imwrite("/home/oraby/Pictures/presentation/lane_detection/9_final_lanes.png", _main_image[:, :, ::-1])


def filter_lines(lines):
    left_lines, right_lines = [], []
    epsilon = 1e-8  # Add a small epsilon value

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                # if abs(y2 - y1) > abs(x2 - x1) * 0.4:
                slope = (y2 - y1) / (x2 - x1 + epsilon)
                if 0.5 < np.abs(slope) < 2:
                    if slope < 0:
                        left_lines.append(line)
                    else:
                        right_lines.append(line)

    return left_lines, right_lines


def merge_lines(left_lines, right_lines):
    if left_lines:
        left_avg = np.mean(left_lines, axis=0)
    else:
        left_avg = []

    if right_lines:
        right_avg = np.mean(right_lines, axis=0)
    else:
        right_avg = []

    return [left_avg, right_avg]


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img


def vis_lanes(img, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        for x1, y1, x2, y2 in line:
            if not any(np.isnan([x1, y1, x2, y2])):
                cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
