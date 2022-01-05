import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

import camera as camservo
import constant
import picar
from car import drive
from image_processing import preprocess_image
from image_processing import show_image
from picar import back_wheels, front_wheels

camera = cv2.VideoCapture(0)
camera.set(3,__SCREEN_WIDTH )
camera.set(4,__SCREEN_HEIGHT )

# Picar setup todo: refactor
picar.setup()
db_file = constant.DB_FILE
fw = front_wheels.Front_Wheels(debug=False, db=db_file)
bw = back_wheels.Back_Wheels(debug=False, db=db_file)
cam = camservo.Camera(debug=False, db=db_file)
bw.ready()
fw.ready()
cam.ready()


interpreter = tflite.Interpreter(model_path=__MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# my_signature = interpreter.get_signature_runner() todo: figure out signatures

while camera.isOpened():
    _, frame = camera.read()
    processed_frame = preprocess_image(frame)
    show_image("Frame", processed_frame[0], constant.SHOW_IMAGE)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        bw.speed = 0
        bw.stop()
        break

    input_data = np.asarray(processed_frame, dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(int(output_data[0]))
    drive(fw, bw, int(output_data[0]), constant.DEFAULT_SPEED)
camera.release()
cv2.destroyAllWindows()
