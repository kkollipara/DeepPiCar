"""
Program to collect training data.
Control car with Play Station controller and save camera frames
"""
import datetime
import logging
import os
import time

import cv2
import numpy as np
import picar
import pygame as pygame
from picar import back_wheels, front_wheels

from onboard import camera as camservo
from onboard.car import drive
from util import constants
from util.image_processing import show_image


def save_frame(iteration, steering_angle, speed, frame):
    img_name = "%s_%03d_%03d.png" % (str(iteration), steering_angle, speed)
    cv2.imwrite(img_name, frame)


logging.basicConfig(level=logging.INFO)

# Joystick Setup
pygame.init()
clock = pygame.time.Clock()
pygame.joystick.init()
joystick_count = pygame.joystick.get_count()

while (joystick_count < 1):
    logging.info('Waiting for controller')
    time.sleep(5)

joystick = pygame.joystick.Joystick(0)
joystick.init()

# camera setup
camera = cv2.VideoCapture(0)
camera.set(3, constants.SCREEN_WIDTH)
camera.set(4, constants.SCREEN_HEIGHT)

# Picar setup
picar.setup()
db_file = constants.DB_FILE
fw = front_wheels.Front_Wheels(debug=False, db=db_file)
bw = back_wheels.Back_Wheels(debug=False, db=db_file)
cam = camservo.Camera(debug=False, db=db_file)
steering_angle = 90
bw.ready()
fw.ready()
cam.ready()

#Data folder setup
collect_data = False
run_folder = os.path.join(constants.ROOT_FOLDER,
                          constants.DATA_FOLDER + datetime.datetime.now().strftime("%y%m%d_%H%M%S"))

if not os.path.exists(run_folder):
    os.makedirs(run_folder)
os.chdir(run_folder)

i = 0
while camera.isOpened():
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.JOYBUTTONDOWN:
            collect_data = not (collect_data)
            logging.info('Data collection: %s' % (collect_data))
            """
            if collect_data:
                joystick.rumble(0.1, 0.5, 200)
            else:
                joystick.rumble(0.1, 0.1, 500)
            """
    steering_angle = int(90 + (joystick.get_axis(2) * 45))
    speed = int(round(joystick.get_axis(1), 1) * -100)
    if constants.IS_CONSTANT_SPEED:
        speed = np.sign(speed) * constants.DEFAULT_SPEED
    drive(fw, bw, steering_angle, speed)

    _, frame = camera.read()
    if speed > 0 and collect_data:
        save_frame(i, steering_angle, speed, frame)
        i += 1

    show_image("Frame", frame, constants.SHOW_IMAGE)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        bw.speed = 0
        bw.stop()
        break   

    clock.tick(40)  # run @ 40 fps

pygame.quit()
camera.release()
cv2.destroyAllWindows()