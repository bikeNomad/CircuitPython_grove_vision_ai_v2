# SPDX-FileCopyrightText: Copyright (c) 2025 Ned Konz for Metamagix
#
# SPDX-License-Identifier: MIT
"""
`grove_vision_ai_v2`
================================================================================

Circuitpython Support for using the Grove Vision AI V2 with pre-built models.

Communication with the Grove Vision AI V2 board is over a UART connection.

Written for AT API "v0", softrware version "2025.01.02"

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
  https://github.com/Seeed-Studio/SSCMA-Micro/blob/1.0.x/docs/protocol/at_protocol.md
* Useful links: https://github.com/djairjr/Seeed-Grove_AI_V2_Dev
* Schematic:
  https://files.seeedstudio.com/wiki/grove-vision-ai-v2/Grove_Vision_AI_Module_V2_Circuit_Diagram.pdf
* 3D printable case:
  https://www.printables.com/model/1250656-3d-printed-foldable-holder-for-grove-vision-ai-mod
"""

# imports
from __future__ import annotations
import time
import json
import binascii
import gc
import busio
from micropython import const

now = time.monotonic

# AT commands from the Arduino library
# Tested
CMD_AT_ACTION = const("ACTION")
CMD_AT_ACTION_STATUS = const("ACTION?")
CMD_AT_ALGOS = const("ALGOS?")
CMD_AT_BREAK = const("BREAK")
CMD_AT_ID = const("ID?")
CMD_AT_INFO = const("INFO?")
CMD_AT_INVOKE = const("INVOKE")
CMD_AT_MODEL = const("MODEL")  # e.g. MODEL=1
CMD_AT_MODELS = const("MODELS?")
CMD_AT_MODEL_STATUS = const("MODEL?")
CMD_AT_NAME = const("NAME?")
CMD_AT_RESET = const("RST")
CMD_AT_SAMPLE = const("SAMPLE")
CMD_AT_SAMPLE_STATUS = const("SAMPLE?")
CMD_AT_SAVE_JPEG = const("save_jpeg()")  # not tested with SD card
CMD_AT_SENSOR = const("SENSOR")
CMD_AT_SENSORS = const("SENSORS?")
CMD_AT_SENSOR_STATUS = const("SENSOR?")
CMD_AT_STATUS = const("STAT?")
CMD_AT_VERSION = const("VER?")

# Recognized but don't know args yet
CMD_AT_LED = const("led")

# 'name' values in event responses
EVENT_INVOKE = const("INVOKE")
EVENT_SAMPLE = const("SAMPLE")
EVENT_SUPERVISOR = const("SUPERVISOR")

# 'type' values in responses
CMD_TYPE_RESPONSE = const(0)
CMD_TYPE_EVENT = const(1)
CMD_TYPE_LOG = const(2)

# 'code' values in responses
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


class DecodeError(ValueError):
    pass


class Perf:
    """Performance metrics from model inference.

    Attributes:
        preprocess: Preprocessing time in milliseconds.
        inference: Inference time in milliseconds.
        postprocess: Postprocessing time in milliseconds.
    """

    def __init__(self, preprocess: int = 0, inference: int = 0, postprocess: int = 0) -> None:
        self.preprocess = preprocess
        self.inference = inference
        self.postprocess = postprocess

    def __repr__(self) -> str:
        return f"Perf(pre={self.preprocess}, inference={self.inference}, post={self.postprocess})"


class Box:
    """Bounding box detection result from object detection models.

    The box is represented with center coordinates (x, y) and dimensions (w, h).

    Attributes:
        x: X-coordinate of box center in pixels (0-240 for default camera resolution).
        y: Y-coordinate of box center in pixels (0-240 for default camera resolution).
        w: Width of box in pixels.
        h: Height of box in pixels.
        score: Confidence score (0-100).
        target: Target class index.
    """

    def __init__(self, x: int, y: int, w: int, h: int, score: int, target: int) -> None:
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.score = score
        self.target = target

    def __repr__(self) -> str:
        return f"Box(x={self.x}, y={self.y}, w={self.w}, h={self.h}, score={self.score}, target={self.target})"

    @property
    def left(self) -> float:
        """Left edge x-coordinate of the box."""
        return self.x - self.w / 2

    @property
    def right(self) -> float:
        """Right edge x-coordinate of the box."""
        return self.x + self.w / 2

    @property
    def top(self) -> float:
        """Top edge y-coordinate of the box."""
        return self.y - self.h / 2

    @property
    def bottom(self) -> float:
        """Bottom edge y-coordinate of the box."""
        return self.y + self.h / 2


class Class:
    """Classification result from image classification models.

    Attributes:
        score: Confidence score (0-100).
        target: Target class index.
    """

    def __init__(self, score: int, target: int) -> None:
        self.score = score
        self.target = target

    def __repr__(self) -> str:
        return f"Class(target={self.target}, score={self.score})"


class Point:
    """Point detection result, used for landmarks or keypoint detection.

    Attributes:
        x: X-coordinate in pixels.
        y: Y-coordinate in pixels.
        score: Confidence score (0-100).
        target: Target point index.
    """

    def __init__(self, x: int, y: int, score: int, target: int) -> None:
        self.x = x
        self.y = y
        self.score = score
        self.target = target

    def __repr__(self) -> str:
        return f"Point(x={self.x}, y={self.y}, score={self.score}, target={self.target})"


class Keypoint:
    """Keypoint detection result combining a bounding box with multiple points.

    Used for pose estimation and similar tasks where objects have multiple landmarks.

    Attributes:
        box: Bounding box containing the detected object.
        points: List of Point objects representing keypoints/landmarks.
    """

    def __init__(self, box: Box, points: list[Point]) -> None:
        self.box = box
        self.points = points

    def __repr__(self) -> str:
        return f"Keypoint(box={repr(self.box)}, points={repr(self.points)})"


class Image:
    """JPEG image data decoded from base64.

    Attributes:
        data: Raw JPEG image bytes.
    """

    def __init__(self, base64: str) -> None:
        """Initialize Image from base64-encoded string.

        Args:
            base64: Base64-encoded JPEG image data from the AI board.
        """
        self.data: bytes = binascii.a2b_base64(base64)


class ATDevice:
    """Main interface to the Grove Vision AI V2 board via UART AT commands.

    This class handles communication with the Grove Vision AI V2 board using
    the AT command protocol over UART at 921600 baud. It provides methods for
    running inference, capturing images, and querying device information.

    Parameters:
        uart: UART bus object for communication.
        response: Last JSON response received from the device.
        debug: Enable debug output for commands and responses.
        perf: Performance metrics from last inference (Perf object).
        boxes: List of Box objects from last detection inference.
        classes: List of Class objects from last classification inference.
        keypoints: List of Keypoint objects from last keypoint inference.
        points: List of Point objects from last point inference.
        image: Image object from last SAMPLE command (if requested).
        response_bufsize: Size of the response buffer in bytes.

    Example:
        >>> import board
        >>> from grove_vision_ai_v2 import ATDevice, CMD_OK
        >>> ai = ATDevice(board.TX, board.RX)
        >>> if ai.invoke(1, True, True) == CMD_OK:
        ...     print(f"Detected {len(ai.boxes)} objects")
    """

    def __init__(
        self, uart_tx, uart_rx, uart_bufsize: int = 1024, bufsize: int = 1024
    ) -> None:
        """Initialize communication with the Grove Vision AI V2 board.

        Args:
            uart_tx: TX pin for UART communication.
            uart_rx: RX pin for UART communication.
            uart_bufsize: Size of UART receiver buffer in bytes. Default 1024.
                          Increase if bytes are being lost.
            bufsize: Size of response buffer in bytes for JSON parsing. Default 1024.
                     Increase if responses are being truncated.
        
        Note:
            Even with adequate buffer sizes,
            using `invoke()` or `sample_image()` to capture images runs
            the risk of data loss, and you may end up with broken JPEG images.
            This is especially likely with an active USB connection.
        """
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
        gc.collect()
        self._response_buffer = bytearray(bufsize)
        self._mv = memoryview(self._response_buffer)
        self._response = None
        self._remaining_bytes = None
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
        self._last_full_command = None
        self._version = None

    @property
    def response_bufsize(self) -> int:
        return len(self._response_buffer)

    @response_bufsize.setter
    def response_bufsize(self, value: int) -> None:
        self._response_buffer = None  # type: ignore
        self._mv = None  # type: ignore
        gc.collect()
        self._response_buffer = bytearray(value)
        self._mv = memoryview(self._response_buffer)

    @property
    def response(self) -> dict | None:
        return self._response

    @property
    def debug(self) -> bool:
        return self._debug

    @debug.setter
    def debug(self, value: bool) -> None:
        self._debug = value

    @property
    def perf(self) -> Perf:
        return self._perf

    @property
    def boxes(self) -> list[Box]:
        return self._boxes

    @property
    def classes(self) -> list[Class]:
        return self._classes

    @property
    def keypoints(self) -> list[Keypoint]:
        return self._keypoints

    @property
    def points(self) -> list[Point]:
        return self._points

    @property
    def image(self) -> Image | None:
        return self._image

    def _send_command(self, command: str, tag: str | None = None) -> None:
        if tag:
            full_command = f"AT+{tag}@{command}\r\n"
        else:
            full_command = f"AT+{command}\r\n"
        if self.debug:
            print(f"=> {full_command}")
        full_command_bytes = full_command.encode("utf-8")
        self._last_full_command = full_command_bytes
        self.uart.write(full_command_bytes)

    def _retry_command(self) -> None:
        self.uart.write(self._last_full_command)

    def _fetch_response(self, timeout: float) -> str | None:
        """Receive and return the next full JSON response as a string.

        Handles buffering of multiple JSON responses and caches remaining data
        for subsequent calls.

        Args:
            timeout: Maximum time to wait for response in seconds.

        Returns:
            JSON response string, or None on timeout.
        """
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
                wait_for_first = (first_byte_time - t_start) * 1000 if first_byte_time else 0
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

    def _parse_event(self, response: dict) -> None:
        """Handle a JSON event response (type=CMD_TYPE_EVENT).

        Extracts inference results (boxes, classes, points, keypoints, images)
        from INVOKE and SAMPLE events and updates instance attributes.

        Args:
            response: Parsed JSON response dictionary.
        """
        if response["name"] not in (CMD_AT_INVOKE, CMD_AT_SAMPLE):
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

    def _parse_log(self, response: dict) -> None:
        """Handle a log JSON response (type=CMD_TYPE_LOG)."""
        # print(response)
        pass

    def _wait(self, response_type: int, cmd: str, timeout: float = 1.0) -> int:
        end_time = now() + timeout
        while now() < end_time:
            resp = self._fetch_response(timeout)
            if resp is None:
                continue
            response = self._response = self._parse_json(resp)

            retval: int = response["code"]
            if response["type"] == CMD_TYPE_EVENT:
                self._parse_event(response)
            elif response["type"] == CMD_TYPE_LOG:
                self._parse_log(response)
                return retval

            # Get the command up to the first "="
            base_cmd = cmd.split("=")[0]
            if response["type"] == response_type and response["name"] == base_cmd:
                return retval
            # else discard this reply

        return CMD_ETIMEDOUT

    def _flush_serial(self) -> str:
        resp = ""
        while self.uart.in_waiting > 0:
            data = self.uart.read()
            if data:
                resp += data.decode("utf-8")
        return resp

    def _parse_json(self, response: str) -> dict:
        try:
            return json.loads(response)
        except ValueError as exc:
            raise DecodeError(f"Failed to decode JSON response {response}") from exc

    def invoke(self, times: int, diffonly: bool, resultonly: bool, timeout: float = 0.1) -> int:
        """Run inference on the loaded model.

        Results are stored in instance attributes (boxes, classes, points, keypoints, perf).

        Args:
            times: Number of times to run inference (usually 1).
            diffonly: If True, only return results that differ from previous inference.
            resultonly: If True, only return inference results without image data.
            timeout: Maximum time to wait for results in seconds. Default 0.1.

        Returns:
            CMD_OK (0) on success, or error code (CMD_ETIMEDOUT, etc.).

        Example:
            >>> if ai.invoke(1, True, True) == CMD_OK:
            ...     for box in ai.boxes:
            ...         print(f"Detected object at ({box.x}, {box.y})")
        """
        self._send_command(f"{CMD_AT_INVOKE}={times},{int(diffonly)},{int(resultonly)}")
        if (err := self._wait(CMD_TYPE_RESPONSE, CMD_AT_INVOKE, 0.05)) == CMD_OK:
            return self._wait(CMD_TYPE_EVENT, CMD_AT_INVOKE, timeout)
        return err

    def sample_image(self, times: int = 1, timeout: float = 0.1) -> int:
        """Capture an image from the camera without running inference.

        The image data is stored in the `image` attribute as an Image object.

        Args:
            times: Number of images to capture (usually 1).
            timeout: Maximum time to wait for image capture in seconds. Default 0.1.

        Returns:
            CMD_OK (0) on success, or error code (CMD_ETIMEDOUT, etc.).

        Example:
            >>> if ai.sample_image() == CMD_OK and ai.image:
            ...     with open('capture.jpg', 'wb') as f:
            ...         f.write(ai.image.data)
        """
        self._send_command(f"{CMD_AT_SAMPLE}={times}")
        if (err := self._wait(CMD_TYPE_RESPONSE, CMD_AT_SAMPLE, 0.05)) == CMD_OK:
            return self._wait(CMD_TYPE_EVENT, CMD_AT_SAMPLE, timeout)
        return err

    def id(self, cache: bool = True) -> str | None:
        """Get the device ID.

        Args:
            cache: If True, return cached value if available. Default True.

        Returns:
            Device ID string, or None on error.
        """
        if cache and self._id:
            return self._id

        self._send_command(CMD_AT_ID)
        if self._wait(CMD_TYPE_RESPONSE, CMD_AT_ID) == CMD_OK:
            if self._response:
                self._id = self._response["data"]
                return self._id
        return None

    def name(self, cache: bool = True) -> str | None:
        """Get the device name.

        Args:
            cache: If True, return cached value if available. Default True.

        Returns:
            Device name string, or None on error.
        """
        if cache and self._name:
            return self._name

        self._send_command(CMD_AT_NAME)
        if self._wait(CMD_TYPE_RESPONSE, CMD_AT_NAME, 3.0) == CMD_OK:
            if self._response:
                self._name = self._response["data"]
                return self._name
        return None

    def version(self, cache: bool = True) -> dict | None:
        """Get the firmware version information.

        Args:
            cache: If True, return cached value if available. Default True.

        Returns:
            Dictionary with version info including 'at_api', 'software', etc., or None on error.
        """
        if cache and self._version:
            return self._version

        self._send_command(CMD_AT_VERSION)
        if self._wait(CMD_TYPE_RESPONSE, CMD_AT_VERSION) == CMD_OK:
            if self.response:
                self._version = self.response["data"]
                return self._version
        return None

    def at_api(self) -> str | None:
        """Get the AT API version string.

        Returns:
            AT API version string (e.g., "v0"), or None on error.
        """
        ver = self.version()
        if ver:
            return ver["at_api"]
        return None

    def info(self, cache: bool = True) -> str | None:
        """Get the model information as a base64-encoded string.

        Use model_info() to get the decoded model metadata.

        Args:
            cache: If True, return cached value if available. Default True.

        Returns:
            Base64-encoded model info string, or None on error.
        """
        if cache and self._info:
            return self._info

        self._send_command(CMD_AT_INFO)
        if self._wait(CMD_TYPE_RESPONSE, CMD_AT_INFO, 3.0) == CMD_OK:
            if self._response:
                self._info = self._response["data"]["info"]
                return self._info
        return None

    def model_info(self) -> dict | None:
        """
        For a model loaded from the Sensecraft web site, return a dict describing the model.
        
        Example:

        .. code-block:: python

            {
                "author": "SenseCraft AI",
                "model_id": "60086",
                "model_name": "Person Detection--Swift YOLO",
                "model_ai_framwork": "6",
                "checksum": "f2b99229ba108c82de9379c4b6ad6354",
                "arguments": {
                    "createdAt": 1705306231,
                    "size": 1644.08,
                    "task": "detect",
                    "conf": 50,
                    "iou": 45,
                    "updatedAt": 1747633412,
                    "icon": "https://sensecraft-statics.oss-accelerate.aliyuncs.com/refer/pic/1705306138275_iykYXV_detection_person.png",
                    "url": "https://sensecraft-statics.oss-accelerate.aliyuncs.com/refer/model/1705306215159_jVQf4u_swift_yolo_nano_person_192_int8_vela(2).tflite",
                },
                "classes": ["person"],
                "version": "1.0.0",
            }

        """
        inf = self.info()
        if inf is None:
            return None
        try:
            return json.loads(binascii.a2b_base64(inf))
        except ValueError:
            return inf

    def clean_actions(self) -> int:
        """Clear all configured actions.

        Returns:
            CMD_OK (0) on success, or error code.
        """
        self._send_command(f'{CMD_AT_ACTION}=""')
        return self._wait(CMD_TYPE_RESPONSE, CMD_AT_ACTION)

    def save_jpeg(self) -> int:
        """Configure the board to save captured images to SD card.

        Requires an SD card to be mounted on the AI board. Use clean_actions()
        to disable this feature.

        Returns:
            CMD_OK (0) on success, or error code.
        """
        self._send_command(f'{CMD_AT_ACTION}="{CMD_AT_SAVE_JPEG}"')
        return self._wait(CMD_TYPE_RESPONSE, CMD_AT_ACTION)

    def perform_command(self, cmd: str, tag: str | None = None) -> int:
        """Perform a raw AT command.

        The response will be available in self.response.

        Args:
            cmd: Command string (e.g., "MODEL=1", "MODELS?").
            tag: Optional tag for the command. Default None.

        Returns:
            CMD_OK (0) on success, CMD_ETIMEDOUT on timeout, or other error code.

        Example:
            >>> ai.perform_command("MODEL=1")  # Switch to model 1
            >>> print(ai.response)
        """
        retries = 0
        while retries < 2:
            self._send_command(cmd, tag)
            try:
                return self._wait(CMD_TYPE_RESPONSE, cmd)
            # Retry on decode error
            except DecodeError:
                retries += 1
                continue
        return CMD_ETIMEDOUT


__name__ = "grove_vision_ai_v2"
__version__ = "0.1.0"
__repo__ = "https://github.com/bikeNomad/CircuitPython_grove_vision_ai_v2.git"
