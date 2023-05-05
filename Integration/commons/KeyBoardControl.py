import carla
import pygame
from pygame.locals import K_ESCAPE
from pygame.locals import K_SPACE
from pygame.locals import K_a
from pygame.locals import K_d
from pygame.locals import K_p
from pygame.locals import K_q
from pygame.locals import K_s
from pygame.locals import K_w

from commons import CONSTANTS


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
            self._control.throttle = CONSTANTS.MAX_THROTTLE
        else:
            self._control.throttle = 0.0

        if keys[K_s]:
            self._control.brake = CONSTANTS.MAX_BREAK
        else:
            self._control.brake = 0.0

        if keys[K_a]:
            self._control.steer = CONSTANTS.STEER_LEFT
        elif keys[K_d]:
            self._control.steer = CONSTANTS.STEER_RIGHT
        else:
            self._control.steer = 0.0

        self._control.hand_brake = keys[K_SPACE]

    @staticmethod
    def _is_quit_shortcut(key):
        return key == K_ESCAPE
