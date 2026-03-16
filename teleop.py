from typing import Any
from queue import Queue
import threading
import time
import numpy as np
import hid
from pynput import keyboard
from lerobot.utils.errors import (
    DeviceAlreadyConnectedError,
    DeviceNotConnectedError,
)
import ik_ur5e 
from scipy.spatial.transform import Rotation


SIGN_VX  =  1   # forward / back
SIGN_VY  = -1   # left / right
SIGN_VZ  = -1   # up / down
POS_SENSITIVITY = 0.00009   # meters per raw unit per step — tune up/down
DEADZONE = 10   # raw units, not normalized

def _convert_buffer(b1: int, b2: int) -> float:
    """Convert two raw HID bytes to signed value (-350 to 350)."""
    value = b1 | (b2 << 8)
    if value >= 32768:
        value -= 65536
    return float(value)


def tcp_pose_to_matrix(pose) -> np.ndarray:
    """Convert UR5e TCP pose [x,y,z,rx,ry,rz] (rotation vector) to 4x4 matrix."""
    T = np.eye(4)
    T[0:3, 3] = pose[0:3]
    rvec = np.array(pose[3:6])
    angle = np.linalg.norm(rvec)
    if angle > 1e-8:
        T[0:3, 0:3] = Rotation.from_rotvec(rvec).as_matrix()
    return T

class UR5eTeleop:
    def __init__(self, robot):
        self.robot = robot
        self._is_connected   = False
        self._device         = None
        self._device_name    = ""
        self.misc_keys_queue = Queue()
        self.ik = ik_ur5e.IK_UR5e()
        # Gripper
        self._gripper_lock   = threading.Lock()
        self._gripper_open   = True
        self._gripper_state  = 0.0   # 0.0 = open, 1.0 = closed

        # Latest delta from HID thread
        self._delta_lock = threading.Lock()
        self._delta_pos  = np.zeros(3)   # [x, y, z] in meters
        self._hid_thread = None
        self._stop_hid   = threading.Event()

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

        found = False
        for _ in range(5):
            for device in hid.enumerate():
                if device["product_string"] == "SpaceMouse Compact" and device["usage_page"] == 1:
                    found = True
                    self._device = hid.Device(device["vendor_id"], device["product_id"])  # only here
                    self._device_name = device["product_string"]
                    break
            if found:
                break
            time.sleep(1.0)

        if not found:
            raise RuntimeError(
                "SpaceMouse not found. Check USB connection and hidapi permissions.\n"
                "Try: sudo chmod a+rw /dev/hidraw*"
            )

        self._stop_hid.clear()
        self._hid_thread = threading.Thread(target=self._hid_reader, daemon=True)
        self._hid_thread.start()

        self._is_connected = True
        print(f"[INFO] SpaceMouse connected: {self._device_name}")
        print("[INFO] Left button OR keyboard 'g': toggle gripper")
        print("[INFO] Press 'q' or ESC to quit")

    def disconnect(self):
        if not self._is_connected:
            return
        self._stop_hid.set()
        if self._hid_thread is not None:
            self._hid_thread.join(timeout=1.0)
        try:
            self._device.close()
        except Exception:
            pass
        self._is_connected = False

    def _hid_reader(self):
        """Continuously reads raw HID packets in background thread."""
        is_universal = "Universal Receiver" in self._device_name

        while not self._stop_hid.is_set():
            data = self._device.read(13 if is_universal else 7)
            if not data:
                time.sleep(0.001)
                continue

            with self._delta_lock:
                if data[0] == 1:
                    # Translation packet
                    rx = _convert_buffer(data[1], data[2])
                    ry = _convert_buffer(data[3], data[4])
                    rz = _convert_buffer(data[5], data[6])

                    dx = SIGN_VX * (rx if abs(rx) > DEADZONE else 0.0) * POS_SENSITIVITY
                    dy = SIGN_VY * (ry if abs(ry) > DEADZONE else 0.0) * POS_SENSITIVITY
                    dz = SIGN_VZ * (rz if abs(rz) > DEADZONE else 0.0) * POS_SENSITIVITY

                    self._delta_pos = np.array([dx, dy, dz])

                elif data[0] == 3:
                    # Button packet — release lock first to avoid deadlock
                    pass

            # Button handling outside delta_lock
            if data[0] == 3:
                if data[1] == 1:      # left button PRESSED → close
                    self._set_gripper(closed=True)
                elif data[1] == 0:    # left button RELEASED → open
                    self._set_gripper(closed=False)

    def _set_gripper(self, closed: bool):
        with self._gripper_lock:
            self._gripper_open  = not closed
            self._gripper_state = 1.0 if closed else 0.0
        print(f"[GRIPPER] {'CLOSE' if closed else 'OPEN'}")
        
    def get_action(self) -> dict:
        if not self._is_connected:
            raise DeviceNotConnectedError("SpaceMouse not connected")

        # Handle keyboard
        pending = []
        while not self.misc_keys_queue.empty():
            pending.append(self.misc_keys_queue.get_nowait())
        for k in pending:
            if k == "g":
                self._toggle_gripper()
            else:
                self.misc_keys_queue.put(k)

        # Get latest delta from HID thread
        with self._delta_lock:
            delta = self._delta_pos.copy()
        with self._gripper_lock:
            gripper_pos = self._gripper_state

        # Clamp delta for safety
        norm = np.linalg.norm(delta)
        if norm > self.robot.max_step_m:
            delta = delta * (self.robot.max_step_m / norm)

        # Build target TCP pose
        current_pose = self.robot.rtde_r.getActualTCPPose()
        target_pose  = list(current_pose)
        target_pose[0] += delta[0]
        target_pose[1] += delta[1]
        target_pose[2] += delta[2]

        # IK
        current_q    = self.robot.rtde_r.getActualQ()
        target_matrix = tcp_pose_to_matrix(target_pose)
        target_q = self.robot.ik.findClosestIK(target_matrix, current_q)
        if target_q is None:
            target_q = current_q

        # Return joint angles directly — ready for send_action and recording
        action_dict = {f"joint.q{i}": float(target_q[i]) for i in range(6)}
        action_dict["gripper.pos"] = float(gripper_pos)
        return action_dict

    def calibrate(self)  -> None: pass
    def configure(self)        : pass
    def send_feedback(self, feedback: dict[str, Any]) -> None: pass