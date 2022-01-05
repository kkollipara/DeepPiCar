import os
import random
from datetime import datetime

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

from offboard.train_model import load_data
from util import constants
from util.image_processing import my_imread
from util.image_processing import preprocess_image

TEST_SIZE = 2000

lite_model_path = os.path.join(constants.ROOT_FOLDER, constants.MODEL_FOLDER, constants.LITE_MODEL_NAME)
image_paths = load_data(os.path.join(constants.ROOT_FOLDER, constants.DATA_FOLDER))
interpreter = tflite.Interpreter(model_path=lite_model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
image_paths = random.sample(image_paths, TEST_SIZE)

time_1 = datetime.now()
for path in image_paths:
    frame = my_imread(path)
    processed_frame = preprocess_image(frame)
    input_data = np.asarray(processed_frame, dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
time_2 = datetime.now()
time_diff = time_2-time_1
time_diff = time_diff.total_seconds()
fps = TEST_SIZE / time_diff
print('FPS: %s' % fps)

cv2.destroyAllWindows()
