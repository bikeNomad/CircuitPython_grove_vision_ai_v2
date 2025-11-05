# SPDX-FileCopyrightText: Copyright (c) 2025 Ned Konz for Metamagix
#
# SPDX-License-Identifier: MIT
"""
This uses a pre-trained model (people detector or face detector) to
drive a servo motor to point to people.

Copy this file to your CIRCUITPY volume (as main.py) along with the grove_vision_ai_v2 library.
Ensure you've deleted the code.py file.
"""

import time
import board
import pwmio
from adafruit_motor import servo
from micropython import const
from grove_vision_ai_v2 import ATDevice, CMD_OK
from digitalio import DigitalInOut


# Configuration
# Scaling (pixels to degrees)
PIXEL_SCALE = -0.3
HALF_IMAGE_WIDTH = const(120)


# Pin usage (change as necessary)
TX_PIN = board.TX
RX_PIN = board.RX
SERVO_PWM_PIN = board.D0
LED_PIN = board.LED_BLUE

# Smoothing params
SFA = const(0.75)
SFB = const(0.25)
assert 1.0 - (SFA + SFB) < 0.0001

# Globals
now = time.monotonic  # cached
ai = ATDevice(TX_PIN, RX_PIN)
pwm = pwmio.PWMOut(SERVO_PWM_PIN, duty_cycle=2**15, frequency=50)
motor = servo.Servo(pwm)
led = DigitalInOut(LED_PIN)
led.switch_to_output(value=True)

target_angle = 90
motor.angle = target_angle  # [0..180]
smoothed_angle = target_angle


def enable_led(value):
    led.value = not bool(value)

def set_motor(target_angle):
    global smoothed_angle
    if target_angle is not None:
        # exponential smoothing
        smoothed_angle = smoothed_angle * SFA + target_angle * SFB
        motor.angle = smoothed_angle
    # print(f"target: {target_angle}, new: {smoothed_angle}")  # DEBUG


def centroid_to_angle(x):
    return 90 + (x - HALF_IMAGE_WIDTH) * PIXEL_SCALE


def get_best_box_angle(boxes):
    if not boxes or len(boxes) == 0:
        return None
    # Choose the box with the best score
    # boxes_sorted = sorted(boxes, key=lambda x: x.score, reverse=True)
    # Choose the widest box (nearest person)
    boxes_sorted = sorted(boxes, key=lambda x: x.w, reverse=True)
    # Translate from x values 0-240 to angle values 180-0
    return centroid_to_angle(boxes_sorted[0].x)


# Turn on debug to watch communications
# ai.debug = True

# print(f"info={ai.info()}")
# print(f"id={ai.id()}")
# print(f"name={ai.name()}")

while True:
    try:
        started = now()
        err = ai.invoke(1, True, True)
        duration = int((now() - started) * 1000)
        enable_led(False)
        if err != CMD_OK:
            print(f"{duration} No response")
            set_motor(target_angle)
            continue
        best_angle = get_best_box_angle(ai.boxes)
        if best_angle is not None:
            enable_led(True)
            target_angle = best_angle
            print(f"({duration} Boxes: {ai.boxes}, Perf: {ai.perf}")
            set_motor(best_angle)
    except ValueError as e:
        print(e)
    except KeyboardInterrupt:
        motor.angle = 90
        break
