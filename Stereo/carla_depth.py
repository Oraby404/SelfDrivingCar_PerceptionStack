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
import cv2
import numpy as np
import random
from queue import Queue

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
        vehicle_bp.set_attribute('color', '15,150,150')

        vehicle_transform = random.choice(self.world.get_map().get_spawn_points())

        self.vehicle = self.world.spawn_actor(vehicle_bp, vehicle_transform)

        # Set up the camera sensor.
        self.camera_manager = CameraManager(self.vehicle, self._gamma)

    def render(self, display):
        self.camera_manager.render(display)

    def destroy(self):
        if self.camera_manager.left_cam is not None:
            self.camera_manager.left_cam.stop()
            self.camera_manager.left_cam.destroy()

        if self.camera_manager.right_cam is not None:
            self.camera_manager.right_cam.stop()
            self.camera_manager.right_cam.destroy()

        if self.vehicle is not None:
            self.vehicle.destroy()


# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720


class CameraManager(object):
    def __init__(self, parent_actor, gamma_correction):
        self.left_cam = None
        self.right_cam = None

        self.frames_queue = Queue()

        world = parent_actor.get_world()
        blueprint_library = world.get_blueprint_library()

        ######################################################################################

        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(WINDOW_WIDTH))
        camera_bp.set_attribute('image_size_y', str(WINDOW_HEIGHT))
        camera_bp.set_attribute('gamma', str(gamma_correction))
        camera_bp.set_attribute('sensor_tick', str(0.0625))  # 16 frames per second

        left_camera_transform = carla.Transform(carla.Location(x=2, y=-0.2, z=1.5))
        right_camera_transform = carla.Transform(carla.Location(x=2, y=0.2, z=1.5))

        self.left_cam = world.spawn_actor(camera_bp, left_camera_transform, attach_to=parent_actor)
        self.right_cam = world.spawn_actor(camera_bp, right_camera_transform, attach_to=parent_actor)

        self.left_cam.listen(lambda image: self._parse_image(image, "left"))
        self.right_cam.listen(lambda image: self._parse_image(image, "right"))

        ######################################################################################

        # K = [[f, 0, Cu],
        #      [0, f, Cv],
        #      [0, 0, 1]]

        Center_X = int(WINDOW_WIDTH / 2)
        Center_Y = int(WINDOW_HEIGHT / 2)

        self.fov = camera_bp.get_attribute("fov").as_float()  # fov = 90.0
        self.focal = WINDOW_WIDTH / (2.0 * np.tan(self.fov * np.pi / 360.0))
        self.baseline = 0.4

        self.K = np.identity(3)
        self.K[0, 0] = self.K[1, 1] = self.focal
        self.K[0, 2] = Center_X
        self.K[1, 2] = Center_Y

        ######################################################################################

        num_disparities = 8 * 16
        block_size = 11
        min_disparity = 0
        window_size = 8

        # Stereo BM matcher
        self.matcher_BM = cv2.StereoBM_create(numDisparities=num_disparities,
                                              blockSize=block_size)

        # Stereo Semi Global Block Matcher
        self.matcher_SGBM = cv2.StereoSGBM_create(minDisparity=min_disparity,
                                                  numDisparities=num_disparities,
                                                  blockSize=block_size,
                                                  P1=8 * 3 * window_size ** 2,
                                                  P2=32 * 3 * window_size ** 2,
                                                  mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

    def render(self, display):
        _left_image = None
        _right_image = None

        if self.frames_queue.qsize() > 1:
            for _ in range(2):
                frame = self.frames_queue.get()
                if frame[1] == 'right':
                    _right_image = frame[0]
                elif frame[1] == 'left':
                    _left_image = frame[0]

            img_l = cv2.cvtColor(_left_image, cv2.COLOR_BGR2GRAY)
            img_r = cv2.cvtColor(_right_image, cv2.COLOR_BGR2GRAY)

            disp_left = self.matcher_SGBM.compute(img_l, img_r).astype(np.float32) / 16

            # Replace all instances of 0 and -1 disparity with a small minimum value (to avoid div by 0 or negatives)
            disp_left[disp_left == 0] = 0.1
            disp_left[disp_left == -1] = 0.1

            depth_map = self.focal * self.baseline / disp_left

            # depth_map = ((depth_map - depth_map.mean()) / depth_map.std()) * 255
            # depth_map = depth_map.astype(np.uint8)

            # Apply log transformation method
            # Converts the image to a depth map using a logarithmic scale,
            # leading to better precision for small distances at the expense of losing it when further away.
            log_depth_map = np.array((255 / np.log(np.max(depth_map))) * np.log(depth_map), dtype=np.uint8)

            # covert to RGB to display it on pygame
            depth_map_rgb = cv2.cvtColor(log_depth_map, cv2.COLOR_GRAY2RGB)

            surface = pygame.surfarray.make_surface(depth_map_rgb.swapaxes(0, 1))
            display.blit(surface, (0, 0))

    def _parse_image(self, image, side):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))[:, :, :3]

        self.frames_queue.put((array, side))


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

            vehicle_world.render(display)
            pygame.display.flip()

            text = font.render('%3.0f FPS' % clock.get_fps(), True, (0, 255, 0))
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
