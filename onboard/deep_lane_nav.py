import os

import cv2
import numpy as np
import picar
import tflite_runtime.interpreter as tflite
from picar import back_wheels, front_wheels

from onboard import camera as camservo
from onboard.car import drive
from util import constants
from util.image_processing import preprocess_image
from util.image_processing import show_image

camera = cv2.VideoCapture(0)
camera.set(3, constants.SCREEN_WIDTH)
camera.set(4, constants.SCREEN_HEIGHT)

# Picar setup todo: refactor
picar.setup()
db_file = constants.DB_FILE
fw = front_wheels.Front_Wheels(debug=False, db=db_file)
bw = back_wheels.Back_Wheels(debug=False, db=db_file)
cam = camservo.Camera(debug=False, db=db_file)
bw.ready()
fw.ready()
cam.ready()

model_path = os.path.join(constants.ROOT_FOLDER, constants.MODEL_FOLDER, constants.MODEL_FINAL_NAME)
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# my_signature = interpreter.get_signature_runner() todo: figure out signatures

while camera.isOpened():
    _, frame = camera.read()
    processed_frame = preprocess_image(frame)
    show_image("Frame", processed_frame[0], constants.SHOW_IMAGE)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        bw.speed = 0
        bw.stop()
        break

    input_data = np.asarray(processed_frame, dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(int(output_data[0]))
    drive(fw, bw, int(output_data[0]), constants.DEFAULT_SPEED)
camera.release()
cv2.destroyAllWindows()
