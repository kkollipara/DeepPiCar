def drive(fw, bw, steering_angle, speed):
    # adjust out of bound steering angles
    if steering_angle > 135:
        steering_angle = 135
    elif steering_angle < 45:
        steering_angle = 45

    fw.turn(steering_angle)

    if speed > 0:
        bw.speed = speed
        bw.forward()
    elif speed < 0:
        bw.speed = -speed
        bw.backward()
    else:
        bw.speed = 0
        bw.stop()
