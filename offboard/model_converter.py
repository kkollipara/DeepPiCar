# Converts base model to TFlite model
import os

import tensorflow as tf
from tensorflow import keras

from util import constants

model_dir = os.path.join(constants.ROOT_FOLDER, constants.MODEL_FOLDER)
model_path = os.path.join(model_dir, constants.MODEL_FINAL_NAME)
model = keras.models.load_model(model_path)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open(os.path.join(model_dir, 'lane_navigation_lite.tflite'), 'wb') as f:
    f.write(tflite_model)
