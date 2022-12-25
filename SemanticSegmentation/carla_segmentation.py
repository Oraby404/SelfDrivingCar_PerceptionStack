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

from Unet_model import UNet
import torch
import torch.nn.functional as F
from torch.backends import cudnn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

colors_platte = torch.tensor([[0, 0, 0],  # Unlabeled
                              [70, 70, 70],  # Building
                              [100, 40, 40],  # Fence
                              [55, 90, 80],  # Other -> Everything that does not belong to any other category.
                              [220, 20, 60],  # Pedestrian
                              [153, 153, 153],  # Pole
                              [157, 234, 50],  # Roadline
                              [128, 64, 128],  # Road
                              [244, 35, 232],  # Sidewalk
                              [107, 142, 35],  # Vegetation
                              [0, 0, 142],  # Car
                              [102, 102, 156],  # Wall
                              [220, 220, 0],  # Traffic sign
                              # not used in current model
                              [70, 130, 180],  # Sky
                              [81, 0, 81],  # Ground
                              [150, 100, 100],  # Bridge
                              [230, 150, 140],  # RailTrack
                              [180, 165, 180],  # GuardRail
                              [250, 170, 30],  # TrafficLight
                              [110, 190, 160],  # Static
                              [170, 120, 50],  # Dynamic
                              [45, 60, 150],  # Water
                              [145, 170, 100]]  # Terrain
                             , device=device)


def semantic_segmentation(_model, _image):
    # turn off gradient tracking
    with torch.no_grad():
        # read the image and its mask
        _img_normalized = _image.astype("float32") / 255.0

        # add batch channel first
        img_tensor = np.expand_dims(_img_normalized, 0).transpose((0, 3, 1, 2))
        img_tensor = torch.from_numpy(img_tensor).to(device)

        # make the prediction
        pred_mask = torch.argmax(F.softmax(_model(img_tensor), dim=1), dim=1).squeeze(0)
        # map to RGB
        pred_mask = colors_platte[pred_mask]

        return pred_mask.cpu().numpy()


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

        if self.camera_manager.mini_cam is not None:
            self.camera_manager.mini_cam.stop()
            self.camera_manager.mini_cam.destroy()

        if self.vehicle is not None:
            self.vehicle.destroy()


# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

MINI_WINDOW_WIDTH = 320
MINI_WINDOW_HEIGHT = 240


class CameraManager(object):
    def __init__(self, parent_actor, gamma_correction):
        self.sensor = None
        self.mini_cam = None

        self.main_surface = None
        self.mini_surface = None
        self.segment_surface = None

        self._parent = parent_actor
        self.recording = False

        self._main_image = None
        self._mini_view_image = None

        world = self._parent.get_world()
        blueprint_library = world.get_blueprint_library()

        ######################################################################################

        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(WINDOW_WIDTH))
        camera_bp.set_attribute('image_size_y', str(WINDOW_HEIGHT))
        camera_bp.set_attribute('gamma', str(gamma_correction))
        camera_bp.set_attribute('sensor_tick', str(0.05))  # 20 frames per second

        # Focus_length = ImageSizeX / (2 * tan(CameraFOV * π / 360)) # fov = 90.0
        # Center_X = ImageSizeX / 2
        # Center_Y = ImageSizeY / 2

        # K = [[f, 0, Cu],
        #      [0, f, Cv],
        #      [0, 0, 1]]

        camera_transform = carla.Transform(carla.Location(x=-6, y=0.0, z=2.5),
                                           carla.Rotation(pitch=8.0))

        self.sensor = world.spawn_actor(camera_bp, camera_transform, attach_to=self._parent,
                                        attachment_type=carla.AttachmentType.SpringArm)

        self.sensor.listen(lambda image: self._parse_image(image))

        ######################################################################################

        mini_camera_bp = blueprint_library.find('sensor.camera.rgb')
        mini_camera_bp.set_attribute('image_size_x', str(MINI_WINDOW_WIDTH))
        mini_camera_bp.set_attribute('image_size_y', str(MINI_WINDOW_HEIGHT))
        mini_camera_bp.set_attribute('gamma', str(gamma_correction))
        mini_camera_bp.set_attribute('sensor_tick', str(0.05))  # 20 frames per second

        mini_camera_transform = carla.Transform(carla.Location(x=1.3, y=0, z=1.3))
        self.mini_cam = world.spawn_actor(mini_camera_bp, mini_camera_transform, attach_to=self._parent)
        self.mini_cam.listen(lambda image: self._parse_image(image, mini=True))

    def render(self, display, _model):
        if self._main_image is not None:
            self.main_surface = pygame.surfarray.make_surface(self._main_image.swapaxes(0, 1))
            display.blit(self.main_surface, (0, 0))

        if self._mini_view_image is not None:
            self.mini_surface = pygame.surfarray.make_surface(self._mini_view_image.swapaxes(0, 1))
            display.blit(self.mini_surface,
                         (WINDOW_WIDTH - MINI_WINDOW_WIDTH - 75, WINDOW_HEIGHT - MINI_WINDOW_HEIGHT - 50))

            mask = semantic_segmentation(_model, self._mini_view_image)

            self.segment_surface = pygame.surfarray.make_surface(mask.swapaxes(0, 1))
            display.blit(self.segment_surface, (75, WINDOW_HEIGHT - MINI_WINDOW_HEIGHT - 50))

    def _parse_image(self, image, mini=False):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]  # shape = height x width
        array = array[:, :, ::-1]  # convert to RGB

        if not mini:
            self._main_image = array
        else:
            self._mini_view_image = array


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
        model = UNet().to(device)
        model.load_state_dict(torch.load("weights/Unet_model.pth")['net_state'])
        # set model to evaluation mode
        model.eval()
        cudnn.benchmark = True

        ###############################################################

        # sim_world = client.get_world()
        sim_world = client.load_world('Town01')
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

            text = font.render('%3.0f FPS' % clock.get_fps(), True, (255, 255, 255))
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
