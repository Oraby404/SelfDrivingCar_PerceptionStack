# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import gc
import os
import threading

import carla
import pygame
import torch
from torch.backends import cudnn

import lane_detection
import object_detection
from commons import CONSTANTS
from commons.KeyBoardControl import KeyboardControl
from commons.VehicleWorld import VehicleWorld
from yolov7.models.experimental import attempt_load
from commons.CONSTANTS import Color


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
        os.chdir('../ObjectDetection/')
        model = attempt_load('yolov7_weights/yolov7.pt', map_location=CONSTANTS.DEVICE)  # load FP32 model
        # set model to evaluation mode
        model.eval()
        cudnn.benchmark = True

        # Initialize client-server connection & world
        client = carla.Client('localhost', 2000)
        client.set_timeout(20.0)

        # Customize World
        sim_world = client.load_world(CONSTANTS.TOWN)
        sim_world.set_weather(carla.WeatherParameters.CloudySunset)

        # World Settings
        settings = sim_world.get_settings()
        original_settings = sim_world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = CONSTANTS.REFRESH_RATE  # 20 FPS
        sim_world.apply_settings(settings)

        # Traffic Manager
        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)

        # Initialize Controller & Vehicle World
        vehicle_world = VehicleWorld(sim_world)
        controller = KeyboardControl()

        # Pygame settings
        display = pygame.display.set_mode((CONSTANTS.WINDOW_WIDTH, CONSTANTS.WINDOW_HEIGHT),
                                          pygame.HWSURFACE | pygame.DOUBLEBUF)
        font = pygame.font.SysFont(CONSTANTS.SYS_FONT_STYLE, CONSTANTS.SYS_FONT_SIZE)

        ###############################################################

        clock = pygame.time.Clock()
        ctr = -1
        while True:
            # gc.collect()
            ctr += 1
            clock.tick()
            sim_world.tick()
            if controller.parse_events(vehicle_world):
                return

            main_image, depth_map, seg_mask = vehicle_world.render()

            if main_image is not None:
                # PREDICTION FREQUENCY
                if ctr % CONSTANTS.PREDICTION_FREQUENCY == 0:
                    ctr = 0
                    # creating threads
                    thread1 = threading.Thread(target=object_detection.detect,
                                               args=(model, main_image, depth_map, CONSTANTS.WINDOW_WIDTH,))
                    thread2 = threading.Thread(target=lane_detection.estimate_lane_lines,
                                               args=(main_image, seg_mask,))

                    # starting thread 1
                    thread1.start()
                    # starting thread 2
                    thread2.start()

                    # wait until thread 1 is finished
                    thread1.join()
                    # wait until thread 2 is finished
                    thread2.join()

                # Sequential execution
                # object_detection.detect(model, main_image, depth_map, CONSTANTS.WINDOW_WIDTH)
                # lane_detection.estimate_lane_lines(main_image,seg_mask)

                display.blit(pygame.surfarray.make_surface(main_image.swapaxes(0, 1)), (0, 0))
                pygame.display.flip()
                text = font.render('% 3.0f FPS' % clock.get_fps(), True, Color.WHITE.value)
                display.blit(text, text.get_rect())
                pygame.display.update()

    finally:

        if vehicle_world is not None:
            vehicle_world.destroy()
            # Always disable sync mode before the script ends to prevent the server blocking whilst waiting for a tick
            sim_world.apply_settings(original_settings)
            traffic_manager.set_synchronous_mode(False)
        pygame.quit()
        del model
        torch.cuda.empty_cache()


def main():
    try:
        game_loop()

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()
