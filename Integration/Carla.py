# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import gc
import os

import carla
import pygame
import torch
from torch.backends import cudnn

from commons import CONSTANTS
from commons.KeyBoardControl import KeyboardControl
from commons.VehicleWorld import VehicleWorld
from yolov7.models.experimental import attempt_load


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
        # load model
        torch.cuda.empty_cache()
        gc.collect()
        # file = bfs_search_for_file('yolov7.pt')
        model = attempt_load('yolov7_weights/yolov7.pt', map_location=CONSTANTS.DEVICE)  # load FP32 model
        # set model to evaluation mode
        model.eval()
        cudnn.benchmark = True

        ###############################################################

        client = carla.Client('localhost', 2000)
        client.set_timeout(20.0)

        sim_world = client.load_world(CONSTANTS.TOWN)
        original_settings = sim_world.get_settings()
        sim_world.set_weather(carla.WeatherParameters.CloudySunset)

        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)

        settings = sim_world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = CONSTANTS.REFRESH_RATE  # 20 FPS
        sim_world.apply_settings(settings)

        ###############################################################

        display = pygame.display.set_mode((CONSTANTS.WINDOW_WIDTH, CONSTANTS.WINDOW_HEIGHT),
                                          pygame.HWSURFACE | pygame.DOUBLEBUF)
        font = pygame.font.SysFont(CONSTANTS.SYS_FONT_STYLE, CONSTANTS.SYS_FONT_SIZE)

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
