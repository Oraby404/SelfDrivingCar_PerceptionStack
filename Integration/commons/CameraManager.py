import carla
import numpy as np
import pygame

from object_detection import detect
from commons import CONSTANTS


class CameraManager:

    def __init__(self, parent_actor, gamma_correction):

        self.cameras = {
            'main_cam': None,
            'depth_cam': None,
            'segmentation_cam': None
        }
        self.main_surface = None

        self._main_image = None
        self._depth_map = None
        self._seg_mask = None

        self._parent = parent_actor

        world = self._parent.get_world()
        blueprint_library = world.get_blueprint_library()

        ######################################################################################

        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(CONSTANTS.WINDOW_WIDTH))
        camera_bp.set_attribute('image_size_y', str(CONSTANTS.WINDOW_HEIGHT))
        camera_bp.set_attribute('gamma', str(gamma_correction))
        camera_bp.set_attribute('sensor_tick', str(CONSTANTS.REFRESH_RATE))  # 20 frames per second

        camera_transform = carla.Transform(carla.Location(x=1.5, y=0, z=1.5))

        self.cameras['main_cam'] = world.spawn_actor(camera_bp, camera_transform, attach_to=self._parent)

        self.cameras['main_cam'].listen(lambda image: self._parse_image(image))

        ######################################################################################

        depth_bp = blueprint_library.find('sensor.camera.depth')
        depth_bp.set_attribute('image_size_x', str(CONSTANTS.WINDOW_WIDTH))
        depth_bp.set_attribute('image_size_y', str(CONSTANTS.WINDOW_HEIGHT))
        depth_bp.set_attribute('sensor_tick', str(CONSTANTS.REFRESH_RATE))  # 20 frames per second

        self.cameras['depth_cam'] = world.spawn_actor(depth_bp, camera_transform, attach_to=self._parent)

        self.cameras['depth_cam'].listen(lambda image: self._parse_depth(image))

        ######################################################################################

        segmentation_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
        segmentation_bp.set_attribute('image_size_x', str(CONSTANTS.WINDOW_WIDTH))
        segmentation_bp.set_attribute('image_size_y', str(CONSTANTS.WINDOW_HEIGHT))
        segmentation_bp.set_attribute('sensor_tick', str(CONSTANTS.REFRESH_RATE))  # 20 frames per second

        self.cameras['segmentation_cam'] = world.spawn_actor(segmentation_bp, camera_transform, attach_to=self._parent)

        self.cameras['segmentation_cam'].listen(lambda image: self._parse_segmentation(image))

        ######################################################################################

        return

    def _parse_image(self, image) -> None:
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8")) \
                    .reshape((image.height, image.width, -1))[:, :, :3][:, :, ::-1]
        self._main_image = np.ascontiguousarray(array, dtype=np.uint8)
        return

    def _parse_depth(self, image) -> None:
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8")) \
            .reshape((image.height, image.width, -1)) \
            .astype(np.float32)

        # the data is stored as 24-bit int across the RGB channels (8 bit per channel)
        depth = (array[:, :, 2] + array[:, :, 1] * 256 + array[:, :, 0] * 256 * 256)
        # normalize it in the range [0, 1]
        normalized = depth / (256 * 256 * 256 - 1)
        # multiply by the max depth distance to get depth in meters
        self._depth_map = 1000 * normalized
        return

    def _parse_segmentation(self, image) -> None:
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        self._seg_mask = np.ascontiguousarray(array[:, :, 2], dtype=np.uint8)
        return

    def render(self):
        if self._main_image is not None and self._depth_map is not None and self._seg_mask is not None:
            return self._main_image, self._depth_map, self._seg_mask
        else:
            return None,None,None

    def destroy(self) -> None:
        for cam in self.cameras.values():
            if cam is not None:
                cam.stop()
                cam.destroy()
        return


if __name__ == '__main__':
    pass
