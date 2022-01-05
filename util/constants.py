import os
from pathlib import Path

# paths
ROOT_FOLDER = Path(os.getcwd()).parent
DATA_FOLDER = 'data/images'
MODEL_FOLDER = 'data/model'
DB_FILE = "/home/pi/SunFounder_PiCar-V/remote_control/remote_control/driver/config"  # edit this to your calib file

# file names
MODEL_CHECKPOINT_NAME = 'lane_navigation_check.h5'
MODEL_FINAL_NAME = 'lane_navigation_final.h5'
LITE_MODEL_NAME = 'lane_navigation_lite.tflite'

# image processing params
SCREEN_WIDTH = 320
SCREEN_HEIGHT = 240
SHOW_IMAGE = False
CROP_DENOM = 8  # factor for cropping top of image; 2 => removes top half; update this based on camera lens and pos

# car params
DEFAULT_SPEED = 50
IS_CONSTANT_SPEED = True

# model hyper params
LEARNING_RATE = 1e-3
BATCH_SIZE = 200
STEPS_PER_EPOCH = 300
NUM_OF_EPOCHS = 30
VALIDATION_STEPS = 200
