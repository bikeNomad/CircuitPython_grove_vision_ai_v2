# SPDX-FileCopyrightText: Copyright (c) 2025 Ned Konz for Metamagix
#
# SPDX-License-Identifier: MIT
"""Human Follower Example for Grove Vision AI V2.

This example uses a pre-trained object detection model (person or face detector)
to control a servo motor that tracks detected people. The servo pans to follow
the nearest person detected in the camera's field of view.

Hardware Requirements:
    - CircuitPython board (Seeed XIAO or Adafruit QtPy recommended)
    - Grove Vision AI V2 board with camera
    - Servo motor connected to D0
    - LED for visual feedback (uses LED_BLUE by default)

Setup:
    1. Flash a person or face detection model from Sensecraft AI to the Grove Vision AI V2
       using its USB-C connector.
       I used this model:
       https://sensecraft.seeed.cc/ai/view-model/60086-person-detection-swift-yolo

    2. Copy `human_follower.mpy` to your `CIRCUITPY/` drive.

    3. Copy `grove_vision_ai_v2.mpy` to `CIRCUITPY/lib/`

    4. Create `CIRCUITPY/code.py` with the following content:

        import human_follower

The servo uses exponential smoothing for smooth motion, and the LED indicates
when a person is detected.
"""

from __future__ import annotations

import time

import board
import pwmio
from adafruit_motor import servo
from digitalio import DigitalInOut
from micropython import const

from grove_vision_ai_v2 import CMD_OK, ATDevice, Box

# Configuration
# Scaling (pixels to degrees)
PIXEL_SCALE = -0.3
HALF_IMAGE_WIDTH = const(120)


# Pin usage (change as necessary)
TX_PIN = board.TX
RX_PIN = board.RX
SERVO_PWM_PIN = board.D0
LED_PIN = board.D1  # active low

SMOOTH_ALPHA = const(0.25)

# Globals
now = time.monotonic  # cached
ai = ATDevice(TX_PIN, RX_PIN)
pwm = pwmio.PWMOut(SERVO_PWM_PIN, duty_cycle=2**15, frequency=50)
motor = servo.Servo(pwm)
led = DigitalInOut(LED_PIN)
led.switch_to_output(value=True)
target_angle = 90
motor.angle = target_angle  # [0..180]


class Smoother:
    """A simple exponential smoother for numerical values.

    Attributes:
        value: The current smoothed value.
        alpha: The smoothing factor (0.0 to 1.0). Higher alpha means less smoothing.
    """

    def __init__(self, initial_value: float, alpha: float) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("Alpha must be between 0.0 and 1.0")
        self.value = initial_value
        self.alpha = alpha

    def update(self, new_value: float) -> float:
        """Update the smoothed value with a new input.

        Args:
            new_value: The new raw value to incorporate into the smoothing.

        Returns:
            The newly smoothed value.
        """
        self.value = self.alpha * new_value + (1.0 - self.alpha) * self.value
        return self.value


smoothed_angle = Smoother(target_angle, SMOOTH_ALPHA)


def enable_led(value: bool) -> None:
    """Enable or disable the indicator LED.

    Args:
        value: True to turn LED on, False to turn off.
            Note: LED logic is inverted (low = on).
    """
    led.value = not bool(value)


def set_motor(target_angle: float | None) -> None:
    """Set the servo motor angle with exponential smoothing.

    Args:
        target_angle: Desired servo angle in degrees (0-180), or None to skip update.
    """
    if target_angle is not None:
        motor.angle = smoothed_angle.update(target_angle)


def centroid_to_angle(x: int) -> float:
    """Convert image x-coordinate to servo angle.

    Maps camera x-coordinate (0-240 pixels) to servo angle (0-180 degrees).
    The center of the image (120 pixels) maps to 90 degrees (servo center).

    Args:
        x: X-coordinate in pixels (0-240).

    Returns:
        Servo angle in degrees (0-180).
    """
    return 90 + (x - HALF_IMAGE_WIDTH) * PIXEL_SCALE


def get_best_box_angle(boxes: list[Box]) -> float | None:
    """Select the best detection box and convert to servo angle.

    Chooses the widest detection box (assumed to be the nearest person)
    and returns the servo angle to center on it.

    Args:
        boxes: List of Box objects from AI inference.

    Returns:
        Servo angle in degrees, or None if no boxes detected.
    """
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
        if err != CMD_OK:
            enable_led(False)
            print(f"{duration} No response")
            set_motor(target_angle)
            continue
        best_angle = get_best_box_angle(ai.boxes)
        if best_angle is not None:
            enable_led(True)
            target_angle = best_angle
            print(f"{duration} Boxes: {ai.boxes}, Perf: {ai.perf}")
            set_motor(best_angle)
        else:
            enable_led(False)
            print(f"{duration} No boxes")
            set_motor(target_angle)
    except ValueError as e:
        print(e)
    except KeyboardInterrupt:
        motor.angle = 90
        break
