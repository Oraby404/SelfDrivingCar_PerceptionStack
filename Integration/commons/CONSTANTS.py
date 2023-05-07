from enum import Enum

# Pygame Settings
SYS_FONT_STYLE = 'Verdana'
SYS_FONT_SIZE = 20

# Simulation Settings
TOWN = "Town10HD"

# CameraManager
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
FRAMES_PER_SECOND = 20
REFRESH_RATE = 1 / FRAMES_PER_SECOND

# KeyboardControl
MAX_THROTTLE = 0.75
MAX_BREAK = 0.75
STEER_LEFT = -0.4
STEER_RIGHT = 0.4

# VehicleWorld
GAMMA = float(2.2)
VEHICLE_TYPE = 'vehicle.dodge.charger_police_2020'
VEHICLE_COLOR = '150,150,150'

# Device
DEVICE = "cuda"


# Colors
class Color(Enum):
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    YELLOW = (255, 255, 0)
    MAGENTA = (255, 0, 255)
    CYAN = (0, 255, 255)
    WHITE = (255, 255, 255)
