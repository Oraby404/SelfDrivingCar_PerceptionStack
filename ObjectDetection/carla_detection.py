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


def detect(model, input_image, imgsz):
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

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], input_image.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, input_image, label=label, color=colors[int(cls)], line_thickness=1)

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
        # vehicle_bp = blueprint_library.find('vehicle.dodge.charger_police_2020')
        vehicle_bp = blueprint_library.find('vehicle.ford.mustang')
        vehicle_bp.set_attribute('color', '10,10,10')

        vehicle_transform = random.choice(self.world.get_map().get_spawn_points())

        self.vehicle = self.world.spawn_actor(vehicle_bp, vehicle_transform)

        # Set up the camera sensor.
        self.camera_manager = CameraManager(self.vehicle, self._gamma)

    def render(self, display, _model):
        self.camera_manager.render(display, _model)

    def destroy(self):
        if self.camera_manager.sensor is not None:
            self.camera_manager.sensor.stop()
            self.camera_manager.sensor.destroy()

        if self.vehicle is not None:
            self.vehicle.destroy()


# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720


class CameraManager(object):
    def __init__(self, parent_actor, gamma_correction):
        self.sensor = None
        self.main_surface = None
        self._main_image = None

        self._parent = parent_actor
        self.recording = False

        world = self._parent.get_world()
        blueprint_library = world.get_blueprint_library()

        ######################################################################################

        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(WINDOW_WIDTH))
        camera_bp.set_attribute('image_size_y', str(WINDOW_HEIGHT))
        camera_bp.set_attribute('gamma', str(gamma_correction))
        camera_bp.set_attribute('sensor_tick', str(0.05))  # 20 frames per second

        camera_transform = carla.Transform(carla.Location(x=1.5, y=0, z=1.5))

        self.sensor = world.spawn_actor(camera_bp, camera_transform, attach_to=self._parent)

        self.sensor.listen(lambda image: self._parse_image(image))

        # image_queue = queue.Queue()
        # camera.listen(image_queue.put)
        # while True:
        #     world.tick()
        #     image = image_queue.get()

    def render(self, display, _model):
        if self._main_image is not None:
            result = detect(_model, self._main_image, WINDOW_WIDTH)

            self.main_surface = pygame.surfarray.make_surface(result.swapaxes(0, 1))
            display.blit(self.main_surface, (0, 0))

    def _parse_image(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]  # shape = height x width
        array = array[:, :, ::-1]  # convert to RGB

        self._main_image = np.ascontiguousarray(array, dtype=np.uint8)
        # self._main_image = array


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
