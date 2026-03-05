from typing import Any
from queue import Queue
import threading
import time
import numpy as np
import pyspacemouse
from pynput import keyboard
from lerobot.utils.errors import (
    DeviceAlreadyConnectedError,
    DeviceNotConnectedError,
)
SIGN_VX  =  1
SIGN_VY  = -1
SIGN_VZ  =  1
SIGN_VRX =  1
SIGN_VRY =  1
SIGN_VRZ =  1
TRANSLATION_SPEED_LIMIT = 0.8
DEADZONE = 0.15


class UR5eTeleop:
    def __init__(self, robot):
        self.robot = robot
        self._is_connected   = False
        self._device         = None
        self.misc_keys_queue = Queue()

        self._gripper_lock   = threading.Lock()
        self._gripper_open   = True
        self._gripper_state  = 0.0   # 0.0=open, 1.0=closed
        self._btn_prev       = False  # previous button state for edge detection

        self._start_keyboard_listener()

    def _start_keyboard_listener(self):
        def on_press(key):
            try:
                if key.char in ["n", "q", "s", "r", "g"]:
                    self.misc_keys_queue.put(key.char)
            except AttributeError:
                if key == keyboard.Key.esc:
                    self.misc_keys_queue.put("esc")
        listener = keyboard.Listener(on_press=on_press)
        listener.daemon = True
        listener.start()

    @property
    def is_connected(self):
        return self._is_connected

    @property
    def is_calibrated(self):
        return True

    def connect(self):
        if self._is_connected:
            raise DeviceAlreadyConnectedError("SpaceMouse already connected")

        self._device = pyspacemouse.open()
        if self._device is None:
            raise RuntimeError("Failed to open SpaceMouse")

        self._is_connected = True
        print("[INFO] SpaceMouse connected")
        print("[INFO] Left button OR keyboard 'g': toggle gripper")
        print("[INFO] Press 'q' or ESC to quit")

    def disconnect(self):
        if not self._is_connected:
            return
        try:
            self._device.close()
        except Exception:
            pass
        self._is_connected = False

    def _toggle_gripper(self):
        with self._gripper_lock:
            self._gripper_open  = not self._gripper_open
            self._gripper_state = 0.0 if self._gripper_open else 1.0
        print(f"[GRIPPER] {'OPEN' if self._gripper_open else 'CLOSE'}")

    def _apply_deadzone(self, value: float) -> float:
        """Deadzone with rescaling so motion starts from 0 (no jump)."""
        if abs(value) < DEADZONE:
            return 0.0
        sign = 1.0 if value > 0 else -1.0
        return sign * (abs(value) - DEADZONE) / (1.0 - DEADZONE)

    def get_action(self) -> dict:
        if not self._is_connected:
            raise DeviceNotConnectedError("SpaceMouse not connected")

        pending = []
        while not self.misc_keys_queue.empty():
            pending.append(self.misc_keys_queue.get_nowait())
        for k in pending:
            if k == "g":
                self._toggle_gripper()
            else:
                self.misc_keys_queue.put(k)

        state = self._device.read()

        btn_now = bool(state.buttons[0]) if state is not None and state.buttons else False
        if btn_now and not self._btn_prev:   # rising edge only → one toggle per press
            self._toggle_gripper()
        self._btn_prev = btn_now

        if state is not None:
            vx  = SIGN_VX  * self._apply_deadzone(state.y)  * TRANSLATION_SPEED_LIMIT
            vy  = SIGN_VY  * self._apply_deadzone(state.x)  * TRANSLATION_SPEED_LIMIT
            vz  = SIGN_VZ  * self._apply_deadzone(state.z)  * TRANSLATION_SPEED_LIMIT
            
        else:
            vx = vy = vz = 0.0

        with self._gripper_lock:
            gripper_pos = self._gripper_state

        return {
            "vx": float(vx), "vy": float(vy), "vz": float(vz),
            "gripper.pos": float(gripper_pos),
        }

    def calibrate(self)  -> None: pass
    def configure(self)        : pass
    def send_feedback(self, feedback: dict[str, Any]) -> None: pass


import logging
import signal
import sys
import config_ur5e
import ur5e
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("teleop_test")


def main():
    robot = ur5e.UR5e(
        config_ur5e.UR5eConfig(
            cameras={
                "scene": OpenCVCameraConfig(index_or_path=0, width=432, height=240, fps=15),
            },
        )
    )
    teleop = UR5eTeleop(robot)
    action = {k: 0.0 for k in ["vx", "vy", "vz", "gripper.pos"]}

    def signal_handler(sig, frame):
        stop = {k: 0.0 for k in ["vx", "vy", "vz"]}
        stop["gripper.pos"] = action.get("gripper.pos", 0.0)
        robot.send_action(stop)
        robot.disconnect()
        teleop.disconnect()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        robot.connect()
        teleop.connect()
        while True:
            t_start = time.monotonic()

            if not teleop.misc_keys_queue.empty():
                key = teleop.misc_keys_queue.queue[0]
                if key in ["q", "esc"]:
                    teleop.misc_keys_queue.get()
                    logger.info("Exit key pressed.")
                    break

            action = teleop.get_action()
            robot.send_action(action)

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        logger.info("Stopping robot...")
        stop = {k: 0.0 for k in ["vx", "vy", "vz"]}
        stop["gripper.pos"] = action.get("gripper.pos", 0.0)
        robot.send_action(stop)
        robot.disconnect()
        teleop.disconnect()
        logger.info("Done.")


if __name__ == "__main__":
    main()