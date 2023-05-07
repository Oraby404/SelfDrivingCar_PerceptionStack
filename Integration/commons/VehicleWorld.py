import random

from commons import CONSTANTS
from commons.CameraManager import CameraManager


class VehicleWorld:

    def __init__(self, carla_world):
        self.world = carla_world

        self.vehicle = None
        self.camera_manager = None

        self._gamma = CONSTANTS.GAMMA

        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.find(CONSTANTS.VEHICLE_TYPE)
        vehicle_bp.set_attribute('color', CONSTANTS.VEHICLE_COLOR)

        vehicle_transform = random.choice(self.world.get_map().get_spawn_points())

        self.vehicle = self.world.spawn_actor(vehicle_bp, vehicle_transform)

        # Set up the camera sensor.
        self.camera_manager = CameraManager(self.vehicle, self._gamma)

    def render(self):
        return self.camera_manager.render()

    def destroy(self):
        if self.camera_manager is not None:
            self.camera_manager.destroy()

        if self.vehicle is not None:
            self.vehicle.destroy()


if __name__ == '__main__':
    pass
