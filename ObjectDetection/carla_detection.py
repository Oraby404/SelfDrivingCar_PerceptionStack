"""

Use WASD keys for control.

    W            : throttle
    S            : brake
    A/D          : steer left/right
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot

    ESC          : quit
"""

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import carla
import numpy as np
import random
import gc
import cv2

import pygame
from pygame.locals import K_ESCAPE
from pygame.locals import K_SPACE
from pygame.locals import K_p
from pygame.locals import K_w
from pygame.locals import K_s
from pygame.locals import K_a
from pygame.locals import K_d
from pygame.locals import K_q

# ==============================================================================
# -- ModelPrediction -----------------------------------------------------------
# ==============================================================================

import torch
from torch.backends import cudnn

from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import letterbox
from yolov7.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov7.utils.plots import plot_one_box

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
#  'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
#  'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
#  'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
#  'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
#  'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
#  'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
#  'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
#  'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
#  'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
#  'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

focus_labels = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'train', 'truck',
                'traffic light', 'stop sign']


def detect(model, input_image, xyz_3D, imgsz):
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

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
                    x_box = x_3D[y_min:y_max, x_min:x_max]
                    y_box = y_3D[y_min:y_max, x_min:x_max]
                    z_box = z_3D[y_min:y_max, x_min:x_max]

                    distance = np.sqrt(x_box ** 2 + y_box ** 2 + z_box ** 2)
                    min_distance = np.min(distance)

                    label = f'{names[int(cls)]} {conf:.2f} {min_distance:.2f}' + 'm'

                    plot_one_box(c1, c2, input_image, label=label, color=colors[int(cls)], line_thickness=1)

                    if conf > 0.9:
                        if names[int(cls)] == 'car':
                            if min_distance < 10:
                                print("Slow down!A Car Ahead of You.")
                        elif names[int(cls)] == 'stop sign':
                            print("Stop Sign Detected!")
                        elif names[int(cls)] == 'traffic light':
                            if min_distance < 15:
                                mid_x = (x_min + x_max) // 2
                                third_height = (y_max - y_min) // 3

                                red_spot = input_image[y_min:y_min + third_height, mid_x - 5:mid_x + 5, 0]
                                green_spot = input_image[y_min + 2 * third_height:y_max, mid_x - 5:mid_x + 5, 1]

                                if np.count_nonzero(red_spot > 230) > 150:
                                    print("RED")
                                elif np.count_nonzero(green_spot > 230) > 150:
                                    print("GREEN")

    return input_image


# ==============================================================================
# -- VehicleWorld --------------------------------------------------------------
# ==============================================================================


class VehicleWorld(object):
    def __init__(self, carla_world):
        self.world = carla_world

        self.vehicle = None
        self.camera_manager = None

        self._gamma = float(2.2)

        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.dodge.charger_police_2020')
        vehicle_bp.set_attribute('color', '150,150,150')

        vehicle_transform = random.choice(self.world.get_map().get_spawn_points())

        self.vehicle = self.world.spawn_actor(vehicle_bp, vehicle_transform)

        # Set up the camera sensor.
        self.camera_manager = CameraManager(self.vehicle, self._gamma)

    def render(self, display, _model):
        self.camera_manager.render(display, _model)

    def destroy(self):
        if self.camera_manager.main_cam is not None:
            self.camera_manager.main_cam.stop()
            self.camera_manager.main_cam.destroy()

        if self.camera_manager.depth_cam is not None:
            self.camera_manager.depth_cam.stop()
            self.camera_manager.depth_cam.destroy()

        if self.vehicle is not None:
            self.vehicle.destroy()


# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720


class CameraManager(object):
    def __init__(self, parent_actor, gamma_correction):
        self.main_cam = None
        self.depth_cam = None

        self.main_surface = None

        self._main_image = None
        self._depth_map = None

        self._parent = parent_actor

        world = self._parent.get_world()
        blueprint_library = world.get_blueprint_library()

        ######################################################################################

        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(WINDOW_WIDTH))
        camera_bp.set_attribute('image_size_y', str(WINDOW_HEIGHT))
        camera_bp.set_attribute('gamma', str(gamma_correction))
        camera_bp.set_attribute('sensor_tick', str(0.05))  # 20 frames per second

        camera_transform = carla.Transform(carla.Location(x=1.5, y=0, z=1.5))

        self.main_cam = world.spawn_actor(camera_bp, camera_transform, attach_to=self._parent)

        self.main_cam.listen(lambda image: self._parse_image(image))

        ######################################################################################

        depth_bp = blueprint_library.find('sensor.camera.depth')
        depth_bp.set_attribute('image_size_x', str(WINDOW_WIDTH))
        depth_bp.set_attribute('image_size_y', str(WINDOW_HEIGHT))
        depth_bp.set_attribute('sensor_tick', str(0.05))  # 20 frames per second

        self.depth_cam = world.spawn_actor(depth_bp, camera_transform, attach_to=self._parent)

        self.depth_cam.listen(lambda image: self._parse_depth(image))

        ######################################################################################

        # Generate a grid of pixel coordinates
        self.u = np.indices((int(WINDOW_HEIGHT), int(WINDOW_WIDTH)))[1]
        self.v = np.indices((int(WINDOW_HEIGHT), int(WINDOW_WIDTH)))[0]

        # K = [[f, 0, Cu],
        #      [0, f, Cv],
        #      [0, 0, 1]]

        self.Center_X = int(WINDOW_WIDTH / 2)
        self.Center_Y = int(WINDOW_HEIGHT / 2)

        self.fov = depth_bp.get_attribute("fov").as_float()  # fov = 90.0
        self.focal = WINDOW_WIDTH / (2.0 * np.tan(self.fov * np.pi / 360.0))
        self.baseline = 0.4

        self.K = np.identity(3)
        self.K[0, 0] = self.K[1, 1] = self.focal
        self.K[0, 2] = self.Center_X
        self.K[1, 2] = self.Center_Y

        self.K_inv = np.linalg.inv(self.K)

        ######################################################################################

    def generate_3D_map(self):
        # Compute 3D x and y coordinates
        x = (self.u - self.Center_X) * self._depth_map / self.focal
        y = (self.v - self.Center_Y) * self._depth_map / self.focal
        z = self._depth_map

        return x, y, z

    def render(self, display, _model):
        if self._main_image is not None and self._depth_map is not None:
            result = detect(_model, self._main_image, self.generate_3D_map(), WINDOW_WIDTH)

            self.main_surface = pygame.surfarray.make_surface(result.swapaxes(0, 1))
            display.blit(self.main_surface, (0, 0))

    def _parse_image(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]  # shape = height x width
        array = array[:, :, ::-1]  # convert to RGB

        self._main_image = np.ascontiguousarray(array, dtype=np.uint8)

    def _parse_depth(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array.astype(np.float32)

        # the data is stored as 24-bit int across the RGB channels (8 bit per channel)
        depth = (array[:, :, 2] + array[:, :, 1] * 256 + array[:, :, 0] * 256 * 256)
        # normalize it in the range [0, 1]
        normalized = depth / (256 * 256 * 256 - 1)
        # multiply by the max depth distance to get depth in meters
        self._depth_map = 1000 * normalized


# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    """Class that handles keyboard input."""

    def __init__(self):
        self._autopilot_enabled = False
        self._control = carla.VehicleControl()

    def parse_events(self, world):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True

                elif event.key == K_q:
                    self._control.gear = 1 if self._control.reverse else -1

                elif event.key == K_p:
                    self._autopilot_enabled = not self._autopilot_enabled
                    world.vehicle.set_autopilot(self._autopilot_enabled)

        if not self._autopilot_enabled:
            self._parse_vehicle_keys(pygame.key.get_pressed())
            self._control.reverse = self._control.gear < 0

            world.vehicle.apply_control(self._control)

    def _parse_vehicle_keys(self, keys):
        if keys[K_w]:
            self._control.throttle = 0.75
        else:
            self._control.throttle = 0.0

        if keys[K_s]:
            self._control.brake = 0.75
        else:
            self._control.brake = 0.0

        if keys[K_a]:
            self._control.steer = -0.4
        elif keys[K_d]:
            self._control.steer = 0.4
        else:
            self._control.steer = 0.0

        self._control.hand_brake = keys[K_SPACE]

    @staticmethod
    def _is_quit_shortcut(key):
        return key == K_ESCAPE


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================

def game_loop():
    pygame.init()
    vehicle_world = None
    sim_world = None
    original_settings = None
    traffic_manager = None

    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(20.0)
        ###############################################################

        # load model
        torch.cuda.empty_cache()
        gc.collect()
        model = attempt_load('yolov7_weights/yolov7.pt', map_location=device)  # load FP32 model
        # set model to evaluation mode
        model.eval()
        cudnn.benchmark = True

        ###############################################################

        # sim_world = client.get_world()
        sim_world = client.load_world('Town10HD')
        original_settings = sim_world.get_settings()
        sim_world.set_weather(carla.WeatherParameters.CloudySunset)

        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)

        settings = sim_world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        sim_world.apply_settings(settings)

        ###############################################################

        display = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
        font = pygame.font.SysFont('Verdana', 20)

        ###############################################################

        vehicle_world = VehicleWorld(sim_world)

        controller = KeyboardControl()

        clock = pygame.time.Clock()
        while True:
            clock.tick()
            sim_world.tick()

            if controller.parse_events(vehicle_world):
                return

            vehicle_world.render(display, model)
            pygame.display.flip()

            text = font.render('% 3.0f FPS' % clock.get_fps(), True, (255, 255, 255))
            display.blit(text, text.get_rect())
            pygame.display.update()

    finally:

        if vehicle_world is not None:
            vehicle_world.destroy()
            # Always disable sync mode before the script ends to prevent the server blocking whilst waiting for a tick
            sim_world.apply_settings(original_settings)
            traffic_manager.set_synchronous_mode(False)

        pygame.quit()


def main():
    try:
        game_loop()

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()
