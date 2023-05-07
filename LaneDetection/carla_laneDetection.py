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


def estimate_lane_lines(segmentation_output):
    # Create an image with pixels belonging to lane boundary categories from the output of semantic segmentation
    lane_mask = np.zeros(segmentation_output.shape, dtype=np.uint8)
    # lane_mask[segmentation_output == 8] = [157, 234, 50]
    lane_mask[segmentation_output == 6] = 255

    # Perform Edge Detection
    edges = cv2.Canny(lane_mask, 50, 150, apertureSize=3)

    # Perform Line estimation
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=100, minLineLength=100, maxLineGap=300)

    lines = np.squeeze(lines)

    # Filter out horizontal lines
    filtered_lines = []
    for line in lines:
        x1, y1, x2, y2 = line
        if abs(y2 - y1) > abs(x2 - x1) * 0.4:
            filtered_lines.append(line)

    filtered_lines = np.squeeze(filtered_lines)

    return filtered_lines


def vis_lanes(image, lane_lines):
    for line in lane_lines:
        x1, y1, x2, y2 = line.astype(int)

        cv2.line(image, (x1, y1), (x2, y2), (255, 0, 255), 2)

    return image


def merge_lane_lines(lines):
    # Step 0: Define thresholds
    slope_similarity_threshold = 0.01
    intercept_similarity_threshold = 40

    # Step 1: Get slope and intercept of lines
    slopes, intercepts = get_slope_intercept(lines)

    clusters = []
    current_inds = []
    itr = 0

    # Step 3: Iterate over all remaining slopes and intercepts and cluster lines that are close to each other using a slope and intercept threshold.
    for slope, intercept in zip(slopes, intercepts):

        exists_in_clusters = np.array([itr in current for current in current_inds])

        if not exists_in_clusters.any():

            slope_cluster = np.logical_and(
                slopes < (slope + slope_similarity_threshold),
                slopes > (slope - slope_similarity_threshold))

            intercept_cluster = np.logical_and(
                intercepts < (intercept + intercept_similarity_threshold),
                intercepts > (intercept - intercept_similarity_threshold))

            inds = np.argwhere(slope_cluster & intercept_cluster)

            # if inds.size:
            #     current_inds.append(inds.flatten)
            #     clusters.append(lines[inds])

            if inds.size:
                current_inds.append(inds)
                cluster_lines = lines[inds]
                cluster_lines_mean = np.mean(cluster_lines, axis=0)
                cluster_lines_mean = np.expand_dims(cluster_lines_mean, axis=0)
                if len(current_inds) == 1:
                    merged_lines = cluster_lines_mean
                else:
                    merged_lines = np.concatenate((merged_lines, cluster_lines_mean), axis=0)

        itr += 1

    # Step 4: Merge all lines in clusters using mean averaging
    # merged_lines = [np.mean(cluster, axis=1) for cluster in clusters]
    merged_lines = np.squeeze(np.array(merged_lines), axis=1)

    return merged_lines


# def get_slope_intercept(lines):
#     slopes = (lines[:, 3] - lines[:, 1]) / (lines[:, 2] - lines[:, 0] + 0.001)
#     intercepts = ((lines[:, 3] + lines[:, 1]) - slopes * (lines[:, 2] + lines[:, 0])) / 2
#     return slopes, intercepts


def get_slope_intercept(lines):
    slopes = []
    intercepts = []
    for line in lines:
        x1, y1, x2, y2 = line
        slope = (y2 - y1) / ((x2 - x1) + 0.001)
        intercept = ((y2 + y1) - slope * (x2 - x1)) / 2
        slopes.append(slope)
        intercepts.append(intercept)
    return slopes, intercepts


def extrapolate_lines(lines, y_min, y_max):
    slopes, intercepts = get_slope_intercept(lines)

    new_lines = []

    for slope, intercept, in zip(slopes, intercepts):
        x1 = (y_min - intercept) / slope
        x2 = (y_max - intercept) / slope
        new_lines.append([x1, y_min, x2, y_max])

    return np.array(new_lines)


#
#
# def find_closest_lines(lines, point):
#     x0, y0 = point
#     distances = []
#     for line in lines:
#         x1, y1, x2, y2 = line
#         distances.append(((x2 - x1) * (y1 - y0) - (x1 - x0) *
#                           (y2 - y1)) / (np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)))
#
#     distances = np.abs(np.array(distances))
#     sorted = distances.argsort()
#
#     return lines[sorted[0:2], :]


# min_y = np.min(np.argwhere(road_mask == 1)[:, 0])
#
# extrapolated_lanes = extrapolate_lines(merged_lane_lines, WINDOW_HEIGHT, min_y)
# final_lanes = find_closest_lines(extrapolated_lanes, dataset_handler.lane_midpoint)
# plt.imshow(dataset_handler.vis_lanes(final_lanes))

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

    def render(self, display):
        self.camera_manager.render(display)

    def destroy(self):
        if self.camera_manager.main_cam is not None:
            self.camera_manager.main_cam.stop()
            self.camera_manager.main_cam.destroy()

        if self.camera_manager.segmentation_cam is not None:
            self.camera_manager.segmentation_cam.stop()
            self.camera_manager.segmentation_cam.destroy()

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
        self.segmentation_cam = None

        self.main_surface = None

        self._main_image = None
        self._seg_mask = None

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

        self.main_cam = world.spawn_actor(camera_bp, camera_transform, attach_to=self._parent)

        self.main_cam.listen(lambda image: self._parse_image(image))

        ######################################################################################

        segmentation_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
        segmentation_bp.set_attribute('image_size_x', str(WINDOW_WIDTH))
        segmentation_bp.set_attribute('image_size_y', str(WINDOW_HEIGHT))
        segmentation_bp.set_attribute('sensor_tick', str(0.05))  # 20 frames per second

        self.segmentation_cam = world.spawn_actor(segmentation_bp, camera_transform, attach_to=self._parent)

        self.segmentation_cam.listen(lambda image: self._parse_segmentation(image))

        ######################################################################################

    def render(self, display):
        if self._main_image is not None and self._seg_mask is not None:
            lane_lines = estimate_lane_lines(self._seg_mask)
            merged_lane_lines = merge_lane_lines(lane_lines)

            out = vis_lanes(self._main_image, merged_lane_lines)

            self.main_surface = pygame.surfarray.make_surface(out.swapaxes(0, 1))

            display.blit(self.main_surface, (0, 0))

    def _parse_image(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]  # shape = height x width
        array = array[:, :, ::-1]  # convert to RGB

        self._main_image = np.ascontiguousarray(array, dtype=np.uint8)

    def _parse_segmentation(self, image):
        # image.convert(carla.ColorConverter.CityScapesPalette)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        # array = array[:, :, :3]  # shape = height x width
        # array = array[:, :, ::-1]  # convert to RGB

        self._seg_mask = np.ascontiguousarray(array[:, :, 2], dtype=np.uint8)


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

            vehicle_world.render(display)
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
