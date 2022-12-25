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
# -- HelperFunctions --------------------------------------------------------------
# ==============================================================================

def FeatureMatcher(descriptor, matcher, first_frame, second_frame, threshold=0.75):
    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_RGB2GRAY)
    second_frame = cv2.cvtColor(second_frame, cv2.COLOR_RGB2GRAY)

    key_points_1, descriptor_1 = descriptor.detectAndCompute(first_frame, None)
    key_points_2, descriptor_2 = descriptor.detectAndCompute(second_frame, None)

    _matches = matcher.knnMatch(descriptor_1, descriptor_2, k=2)

    srcPts = []
    desPts = []

    for m, n in _matches:
        if m.distance < threshold * n.distance:
            srcPts.append(key_points_1[m.queryIdx].pt)  # u1, v1 = kp1[m.queryIdx].pt
            desPts.append(key_points_2[m.trainIdx].pt)  # u2, v2 = kp2[m.trainIdx].pt

    srcPts = np.int32(srcPts)
    desPts = np.int32(desPts)

    return srcPts, desPts


def estimate_motion(frame1_pts, frame2_pts, k):
    # Estimate camera motion between a pair of images

    Essential_matrix, mask = cv2.findEssentialMat(points1=frame1_pts, points2=frame2_pts, cameraMatrix=k,
                                                  method=cv2.RANSAC, prob=0.9)
    rmat_1, rmat_2, tvec = cv2.decomposeEssentialMat(Essential_matrix)

    if np.linalg.det(rmat_1) == 1:
        rmat = rmat_1
    else:
        rmat = rmat_2

    # We select only inlier points
    frame1_pts = frame1_pts[mask.ravel() == 1]
    frame2_pts = frame2_pts[mask.ravel() == 1]

    return rmat, tvec, frame1_pts, frame2_pts


def estimate_trajectory(prev_camera_pose, rmat, tvec):
    """
    Estimate complete camera trajectory from subsequent image pairs

    """
    # Construct the T matrix 4x4
    # Determine current pose from rotation and translation matrices
    current_pose = np.eye(4)
    current_pose[0:3, 0:3] = rmat
    current_pose[0:3, 3] = tvec.T

    # Build the robot's pose from the initial position by multiplying previous and current poses
    new_camera_pose = prev_camera_pose @ np.linalg.inv(current_pose)

    # Calculate current camera position from origin
    position = new_camera_pose @ np.array([0., 0., 0., 1.])

    trajectory = position[0:3]

    return new_camera_pose, trajectory


def visualize_camera_movement(image1, image1_points, image2_points):
    for i in range(0, len(image1_points)):
        # Coordinates of a point on t frame
        p1 = (int(image1_points[i][0]), int(image1_points[i][1]))
        # Coordinates of the same point on t+1 frame
        p2 = (int(image2_points[i][0]), int(image2_points[i][1]))

        cv2.circle(image1, p1, 5, (0, 255, 0), 1)
        cv2.arrowedLine(image1, p1, p2, (0, 255, 0), 1)
        cv2.circle(image1, p2, 5, (255, 0, 0), 1)

    return image1


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
        if self.camera_manager.camera is not None:
            self.camera_manager.camera.stop()
            self.camera_manager.camera.destroy()

        if self.vehicle is not None:
            self.vehicle.destroy()


# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720


class CameraManager(object):
    def __init__(self, parent_actor, gamma_correction):
        self.camera = None
        self.frames_queue = Queue()

        world = parent_actor.get_world()
        blueprint_library = world.get_blueprint_library()

        ######################################################################################

        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(WINDOW_WIDTH))
        camera_bp.set_attribute('image_size_y', str(WINDOW_HEIGHT))
        camera_bp.set_attribute('gamma', str(gamma_correction))
        camera_bp.set_attribute('sensor_tick', str(0.0625))  # 16 frames per second

        camera_transform = carla.Transform(carla.Location(x=2, y=0.2, z=1.5))

        self.camera = world.spawn_actor(camera_bp, camera_transform, attach_to=parent_actor)

        self.camera.listen(lambda image: self._parse_image(image))

        ######################################################################################

        # Generate a grid of pixel coordinates
        self.u = np.indices((int(WINDOW_HEIGHT), int(WINDOW_WIDTH)))[1]
        self.v = np.indices((int(WINDOW_HEIGHT), int(WINDOW_WIDTH)))[0]

        # K = [[f, 0, Cu],
        #      [0, f, Cv],
        #      [0, 0, 1]]

        self.Center_X = int(WINDOW_WIDTH / 2)
        self.Center_Y = int(WINDOW_HEIGHT / 2)

        self.fov = camera_bp.get_attribute("fov").as_float()  # fov = 90.0
        self.focal = WINDOW_WIDTH / (2.0 * np.tan(self.fov * np.pi / 360.0))
        self.baseline = 0.4

        self.K = np.identity(3)
        self.K[0, 0] = self.K[1, 1] = self.focal
        self.K[0, 2] = self.Center_X
        self.K[1, 2] = self.Center_Y

        self.K_inv = np.linalg.inv(self.K)

        ######################################################################################

        # creating SIFT detector
        self.sift = cv2.SIFT_create()

        # creating FLANN matcher
        index_params = dict(algorithm=0, trees=5)
        search_params = dict(checks=50)

        self.flannMatcher = cv2.FlannBasedMatcher(index_params, search_params)

        # Initialize camera pose
        self.camera_pose = np.eye(4)

        ######################################################################################

    def PointCloud_3D(self, depth_map):
        x = (self.u - self.Center_X) * depth_map / self.focal
        y = (self.v - self.Center_Y) * depth_map / self.focal
        p3d = np.stack((x, y, depth_map))

        # p0 = p3d[:, 0] ... # pi = p3d[:, i]
        return p3d

    def PointSet_3D(self, u_coord, v_coord, depth_map):
        # point 2D = [u,v,1] in pixell coordinates
        p2d = np.array([u_coord, v_coord, np.ones_like(u_coord)])

        # P = [X,Y,Z]
        p3d = np.dot(self.K_inv, p2d) * depth_map

        return p3d

    def render(self, display):
        if self.frames_queue.qsize() > 1:
            first_frame = self.frames_queue.get()
            second_frame = self.frames_queue.get()

            srcPts, desPts = FeatureMatcher(self.sift, self.flannMatcher, first_frame, second_frame,
                                            threshold=0.95)

            rmat, tvec, image1_points, image2_points = estimate_motion(srcPts, desPts, self.K)

            image_move = visualize_camera_movement(first_frame, image1_points, image2_points)

            self.camera_pose, trajectory = estimate_trajectory(self.camera_pose, rmat, tvec)

            print(trajectory)

            surface = pygame.surfarray.make_surface(image_move.swapaxes(0, 1))
            display.blit(surface, (0, 0))

    def _parse_image(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))[:, :, :3]
        array = array[:, :, ::-1]  # convert to RGB

        self.frames_queue.put(np.ascontiguousarray(array, dtype=np.uint8))


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
