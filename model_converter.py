import os

import tensorflow as tf
from tensorflow import keras

import constant

model_dir = os.path.join(os.getcwd(), constant.MODEL_FOLDER)
model_path = os.path.join(model_dir, constant.MODEL_FINAL_NAME)
model = keras.models.load_model(model_path)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open(os.path.join(model_dir, 'lane_navigation_lite.tflite'), 'wb') as f:
    f.write(tflite_model)
