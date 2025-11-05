# SPDX-FileCopyrightText: Copyright (c) 2025 Ned Konz for Metamagix
#
# SPDX-License-Identifier: MIT
"""
`grove_vision_ai_v2`
================================================================================

Circuitpython Support for using the Grove Vision AI V2 with pre-built models.

Communication with the Grove Vision AI V2 board is over a UART connection.

* Author(s): Ned Konz

Implementation Notes
--------------------

**Hardware:**

* Seeed Studio Grove Vision AI V2 board:
  https://www.seeedstudio.com/Grove-Vision-AI-Module-V2-p-5851.html

* OV5647 Camera supported by the Grove Vision AI V2 board

* CircuitPython board (the Seeed XIAO series and the Adafruit QtPy boards will plug
  directly onto the Grove Vision AI V2 board)

**Software and Dependencies:**

* Adafruit CircuitPython firmware for the supported boards:
  https://circuitpython.org/downloads

* Adafruit's Motor library: https://github.com/adafruit/Adafruit_CircuitPython_Motor

**References**

* AT Command reference:
  https://github.com/Seeed-Studio/SSCMA-Micro/blob/main/docs/protocol/at-protocol-en_US.md
* Useful links: https://github.com/djairjr/Seeed-Grove_AI_V2_Dev
* Schematic:
  https://files.seeedstudio.com/wiki/grove-vision-ai-v2/Grove_Vision_AI_Module_V2_Circuit_Diagram.pdf
* 3D printable case:
  https://www.printables.com/model/1250656-3d-printed-foldable-holder-for-grove-vision-ai-mod
"""

# imports
import time
import json
import binascii
import gc
import busio
from micropython import const

now = time.monotonic

# AT commands from the Arduino library
# NOTE: most are untetested and may not work.
CMD_AT_ACTION = const("ACTION")
CMD_AT_ALGOS = const("ALGOS")
CMD_AT_BREAK = const("BREAK")
CMD_AT_ID = const("ID?")
CMD_AT_INFO = const("INFO?")
CMD_AT_INVOKE = const("INVOKE")
CMD_AT_LED = const("led")
CMD_AT_MODEL = const("MODEL")
CMD_AT_MODELS = const("MODELS")
CMD_AT_NAME = const("NAME?")
CMD_AT_RESET = const("RST")
CMD_AT_SAMPLE = const("SAMPLE")
CMD_AT_SAVE_JPEG = const("save_jpeg()")
CMD_AT_SENSORS = const("SENSORS")
CMD_AT_STATS = const("STAT")
CMD_AT_TIOU = const("TIOU")
CMD_AT_TSCORE = const("TSCORE")
CMD_AT_VERSION = const("VER?")

EVENT_INVOKE = const("INVOKE")
EVENT_SAMPLE = const("SAMPLE")
EVENT_WIFI = const("WIFI")
EVENT_MQTT = const("MQTT")
EVENT_SUPERVISOR = const("SUPERVISOR")

CMD_TYPE_RESPONSE = const(0)
CMD_TYPE_EVENT = const(1)
CMD_TYPE_LOG = const(2)

CMD_OK = const(0)
CMD_AGAIN = const(1)
CMD_ELOG = const(2)
CMD_ETIMEDOUT = const(3)
CMD_EIO = const(4)
CMD_EINVAL = const(5)
CMD_ENOMEM = const(6)
CMD_EBUSY = const(7)
CMD_ENOTSUP = const(8)
CMD_EPERM = const(9)
CMD_EUNKNOWN = const(10)

RESPONSE_PREFIX = const(b"\r{")
RESPONSE_SUFFIX = const(b"}\n")


class Perf:
    def __init__(self, preprocess=0, inference=0, postprocess=0):
        self.preprocess = preprocess
        self.inference = inference
        self.postprocess = postprocess

    def __repr__(self):
        return f"Perf(pre={self.preprocess}, inference={self.inference}, post={self.postprocess})"


class Box:
    def __init__(self, x, y, w, h, score, target):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.score = score
        self.target = target

    def __repr__(self):
        return (
            f"Box(x={self.x}, y={self.y}, w={self.w}, h={self.h}, score={self.score}, target={self.target})"
        )

    @property
    def left(self):
        return self.x - self.w / 2

    @property
    def right(self):
        return self.x + self.w / 2

    @property
    def top(self):
        return self.y - self.h / 2

    @property
    def bottom(self):
        return self.y + self.h / 2


class Class:
    def __init__(self, score, target):
        self.score = score
        self.target = target

    def __repr__(self):
        return f"Class(target={self.target}, score={self.score})"


class Point:
    def __init__(self, x, y, score, target):
        self.x = x
        self.y = y
        self.score = score
        self.target = target

    def __repr__(self):
        return (
            f"Point(x={self.x}, y={self.y}, score={self.score}, target={self.target})"
        )


class Keypoint:
    def __init__(self, box: Box, points: list[Point]):
        self.box = box
        self.points = points

    def __repr__(self):
        return f"Keypoint(box={repr(self.box)}, points={repr(self.points)})"


class Image:
    def __init__(self, base64):
        self.data = binascii.a2b_base64(base64)


class ATDevice:
    def __init__(self, uart_tx, uart_rx, uart_bufsize=1024, bufsize=8192):
        gc.collect()
        uart = busio.UART(
            uart_tx,
            uart_rx,
            baudrate=921600,
            timeout=0.01,
            receiver_buffer_size=uart_bufsize,
        )
        self.uart = uart
        uart.reset_input_buffer()
        self._response_buffer = bytearray(bufsize)
        self._response = None
        self._remaining_bytes = None
        self._mv = memoryview(self._response_buffer)
        self._debug = False
        self._perf = Perf()
        self._boxes = []
        self._classes = []
        self._keypoints = []
        self._points = []
        self._image = None
        self._id = None
        self._name = None
        self._info = None

    @property
    def response_bufsize(self):
        return len(self._response_buffer)

    @response_bufsize.setter
    def response_bufsize(self, value):
        self._response_buffer = None
        self._mv = None
        gc.collect()
        self._response_buffer = bytearray(value)
        self._mv = memoryview(self._response_buffer)

    @property
    def debug(self):
        return self._debug

    @debug.setter
    def debug(self, value):
        self._debug = value

    @property
    def perf(self):
        return self._perf

    @property
    def boxes(self):
        return self._boxes

    @property
    def classes(self):
        return self._classes

    @property
    def keypoints(self):
        return self._keypoints

    @property
    def points(self):
        return self._points

    @property
    def image(self):
        return self._image

    def _send_command(self, command, tag=None):
        if tag:
            full_command = f"AT+{tag}@{command}\r\n"
        else:
            full_command = f"AT+{command}\r\n"
        if self.debug:
            print(f"=> {full_command}")
        self.uart.write(full_command.encode("utf-8"))

    def _fetch_response(self, timeout):
        """Receive and return the next full JSON response as a string.
        On timeout, return None."""
        t_start = now()
        if self._remaining_bytes:
            response = str(
                self._mv[self._remaining_bytes[0] : self._remaining_bytes[1]], "utf-8"
            ).strip()
            self._remaining_bytes = None
            if self.debug:
                print(f"<=(CACHED) {response} [took {(now() - t_start) * 1000:.1f}ms]")
            return response
        end_time = now() + timeout
        index = 0
        first_byte_time = None
        poll_count = 0
        while now() < end_time:
            poll_count += 1
            if self.uart.in_waiting == 0:
                continue
            if first_byte_time is None:
                first_byte_time = now()
            bytes_read = self.uart.readinto(self._mv[index:])
            index += bytes_read
            if (where := self._response_buffer.find(RESPONSE_SUFFIX, 0, index)) != -1:
                suffix_end = where + 2  # Position after }\n
                response = str(self._mv[0:suffix_end], "utf-8").strip()
                # Check if there are remaining bytes after this response
                if suffix_end < index:
                    self._remaining_bytes = (suffix_end, index)
                    if self.debug:
                        print(f"[BUFFER] Saving {index - suffix_end} remaining bytes")
                else:
                    self._remaining_bytes = None
                elapsed = (now() - t_start) * 1000
                wait_for_first = (
                    (first_byte_time - t_start) * 1000 if first_byte_time else 0
                )
                if self.debug:
                    print(f"<=(NEW) {response}")
                    print(
                        f"[TIMING] _fetch_response: waited {wait_for_first:.1f}ms for first byte, total {elapsed:.1f}ms, {poll_count} polls, {index} bytes"
                    )
                return response
        if self.debug:
            print(
                f"[TIMEOUT] _fetch_response timed out after {(now() - t_start) * 1000:.1f}ms, {poll_count} polls"
            )
        return None

    def _parse_event(self, response: dict):
        """Handle a JSON event response (type=CMD_TYPE_EVENT)."""
        if response["name"] != CMD_AT_INVOKE:
            return

        data = response.get("data", None)
        if data is None:
            return

        if (perf := data.get("perf", None)) and isinstance(perf, list):
            self._perf = Perf(*perf)
        else:
            self._perf = Perf()

        if (boxes := data.get("boxes", None)) and isinstance(boxes, list):
            self._boxes = [Box(*box) for box in boxes]
        else:
            self._boxes.clear()

        if (classes := data.get("classes", None)) and isinstance(classes, list):
            self._classes = [Class(*cls) for cls in classes]
        else:
            self._classes.clear()

        if (points := data.get("points", None)) and isinstance(points, list):
            self._points = [Point(*point) for point in points]
        else:
            self._points.clear()

        self._keypoints.clear()
        if (keypoints := data.get("keypoints", None)) and isinstance(keypoints, list):
            self._keypoints = []
            for kp in keypoints:
                box = Box(*kp[0])
                points = [Point(*point) for point in kp[1]]
                self._keypoints.append(Keypoint(box, points))

        if (image := data.get("image", None)) and isinstance(image, str):
            self._image = Image(image)
        else:
            self._image = None

    def _parse_log(self, response: dict):
        """Handle a log JSON response (type=CMD_TYPE_LOG)."""
        pass

    def _wait(self, type: int, cmd: str, timeout: float = 1.0):
        end_time = now() + timeout
        while now() < end_time:
            resp = self._fetch_response(timeout)
            if resp is None:
                continue
            response = self._response = self._parse_json(resp)
            if response is None:
                continue

            if response["type"] == CMD_TYPE_EVENT:
                self._parse_event(response)
            elif response["type"] == CMD_TYPE_LOG:
                self._parse_log(response)

            retval = response["code"]
            if response["type"] == type and response["name"] == cmd:
                return retval
            # else discard this reply

        return CMD_ETIMEDOUT

    def _flush_serial(self):
        resp = ""
        while self.uart.in_waiting > 0:
            resp += self.uart.read().decode("utf-8")
        return resp

    def _parse_json(self, response):
        try:
            return json.loads(response)
        except ValueError:
            print(f"Failed to decode JSON response {response}")
            return None

    def invoke(self, times: int, diffonly: bool, resultonly: bool, timeout=0.1):
        self._send_command(f"{CMD_AT_INVOKE}={times},{int(diffonly)},{int(resultonly)}")
        if self._wait(CMD_TYPE_RESPONSE, CMD_AT_INVOKE, 0.05) == CMD_OK:
            if self._wait(CMD_TYPE_EVENT, CMD_AT_INVOKE, timeout) == CMD_OK:
                return CMD_OK
        return CMD_ETIMEDOUT

    def id(self, cache=True):
        if cache and self._id:
            return self._id

        self._send_command(CMD_AT_ID)
        if self._wait(CMD_TYPE_RESPONSE, CMD_AT_ID) == CMD_OK:
            self._id = self._response["data"]
            return self._id
        return None

    def name(self, cache=True):
        if cache and self._name:
            return self._name

        self._send_command(CMD_AT_NAME)
        if self._wait(CMD_TYPE_RESPONSE, CMD_AT_NAME, 3.0) == CMD_OK:
            self._name = self._response["data"]
            return self._name
        return None

    def info(self, cache=True):
        if cache and self._info:
            return self._info

        self._send_command(CMD_AT_INFO)
        if self._wait(CMD_TYPE_RESPONSE, CMD_AT_INFO, 3.0) == CMD_OK:
            self._info = self._response["data"]["info"]
            return self._info
        return None

    def clean_actions(self):
        self._send_command(f'{CMD_AT_ACTION}=""')
        if self._wait(CMD_TYPE_RESPONSE, CMD_AT_ACTION) == CMD_OK:
            return CMD_OK
        return CMD_ETIMEDOUT

    def save_jpeg(self):
        self._send_command(f'{CMD_AT_ACTION}="{CMD_AT_SAVE_JPEG}"')
        if self._wait(CMD_TYPE_RESPONSE, CMD_AT_ACTION) == CMD_OK:
            return CMD_OK
        return CMD_ETIMEDOUT

    def perform_command(self, cmd: str, tag: str = None):
        """Perform one of the CMD_AT_xxx commands.
        Return CMD_OK or CMD_ETIMEDOUT."""
        self._send_command(cmd, tag)
        if self._wait(CMD_TYPE_RESPONSE, CMD_AT_ACTION) == CMD_OK:
            return CMD_OK
        return CMD_ETIMEDOUT


__name__ = "grove_vision_ai_v2"
__version__ = "0.0.0+auto.0"
__repo__ = "https://github.com/bikeNomad/CircuitPython_grove_vision_ai_v2.git"
