"""
Use play station controller as remote for picar

"""
import logging
import time

import cv2
import picar
import pygame
from picar import back_wheels, front_wheels

pygame.init()

clock = pygame.time.Clock()
logging.basicConfig(level=logging.INFO)

# Joystick Setup
pygame.joystick.init()
joystick_count = pygame.joystick.get_count()

while (joystick_count < 1):
    logging.info('Waiting for controller')
    time.sleep(5)
    joystick_count = pygame.joystick.get_count()

print("Joystick count: {}".format(joystick_count))
joystick = pygame.joystick.Joystick(0)
joystick.init()
name = joystick.get_name()
print("Joystick name: {}".format(name))
axes = joystick.get_numaxes()
print("Joystick axis count: {}".format(axes))


#Picar setup
picar.setup()
db_file = "/home/pi/SunFounder_PiCar-V/remote_control/remote_control/driver/config"
fw = front_wheels.Front_Wheels(debug=False, db=db_file)
bw = back_wheels.Back_Wheels(debug=False, db=db_file)
bw.ready()
fw.ready()


while True:
    for event in pygame.event.get(): # User did something.
        pass
    axis_lateral = int(90 + (joystick.get_axis(2)*45))
    axis_vertical = int(joystick.get_axis(1)* -100)
    
    print("Lateral axis value: {}".format(axis_lateral))
    print("Vertical axis value: {}".format(axis_vertical))
 
    fw.turn(axis_lateral)

    if (axis_vertical > 0):
        bw.speed = axis_vertical
        bw.forward()
    elif (axis_vertical < 0):
        bw.speed = -axis_vertical
        bw.backward()
    else:
        bw.speed = 0
        bw.stop()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        bw.speed = 0
        bw.stop()
        break

    clock.tick(40)

pygame.quit()