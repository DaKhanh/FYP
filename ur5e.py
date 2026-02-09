import socket
import time
import logging
import numpy as np
from math import ceil

from ur_rtde import rtde_receive
from ur_rtde import dashboard_client

import onRobot.gripper as gripper

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots.robot import Robot
from .config_ur5e import UR5eConfig

logger = logging.getLogger(__name__)


class UR5e(Robot):
    config_class = UR5eConfig
    name = "ur5e"

    def __init__(self, config: UR5eConfig):
        super().__init__(config)

        self.config = config
        self.robot_ip = config.robot_ip
        self.robot_port = config.robot_port

        # RTDE (receive only)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(self.robot_ip)
        self.dash_c = dashboard_client.DashboardClient(self.robot_ip)

        # Gripper
        self.rg6_gripper = gripper.RG6()
        self.last_gripper_state = None

        # Servo socket
        self.conn = None

        # Servo parameters
        self.control_freq = config.control_freq          # e.g. 20 Hz
        self.servo_dt = 1.0 / self.control_freq
        self.servo_lookahead = config.servo_lookahead #0.1
        self.servo_gain = config.servo_gain #300

        # Safety
        self.max_joint_step = config.max_joint_step #0.05  

        # Cameras
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        ft = {}
        for i in range(1, 7):
            ft[f"joint_{i}.pos"] = float
        ft["gripper.pos"] = float
        return ft

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (
                self.config.cameras[cam].height,
                self.config.cameras[cam].width,
                3,
            )
            for cam in self.cameras
        }

    @property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return (
            self.rtde_r.isConnected()
            and self.dash_c.isConnected()
            and all(cam.is_connected for cam in self.cameras.values())
        )

    def connect(self, calibrate: bool = True) -> None:
        self.dash_c.connect()

        self.conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.conn.connect((self.robot_ip, self.robot_port))

        for cam in self.cameras.values():
            cam.connect()

        logger.info("UR5e servo control connected")

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        logger.info("UR5e calibration skipped (assumed calibrated)")

    def configure(self) -> None:
        logger.info("UR5e configuration skipped")

    def disconnect(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None

        for cam in self.cameras.values():
            cam.disconnect()

        logger.info("UR5e disconnected")

    def get_observation(self) -> dict[str, np.ndarray]:
        obs = {}

        q = self.rtde_r.getActualQ()
        for i in range(6):
            obs[f"joint_{i+1}.pos"] = np.float32(q[i])

        obs["gripper.pos"] = np.float32(self.rg6_gripper.get_rg_width())

        for cam_name, cam in self.cameras.items():
            obs[cam_name] = cam.async_read()

        return obs

    def send_action(self, action_list):
        """
        action_list: [j1, j2, j3, j4, j5, j6, gripper]
        """
        action_dict = {
            f"joint_{i+1}.pos": float(action_list[i]) for i in range(6)
        }
        action_dict["gripper.pos"] = float(action_list[6])

        self._write_to_motors(action_dict)


    def _write_to_motors(self, action):
        if self.conn is None:
            return

        # Desired joint positions
        q_des = np.array(
            [float(action[f"joint_{i+1}.pos"]) for i in range(6)],
            dtype=np.float32,
        )

        # Current joint positions
        q_curr = np.array(self.rtde_r.getActualQ(), dtype=np.float32)

        # Clip for safety
        q_des = np.clip(
            q_des,
            q_curr - self.max_joint_step,
            q_curr + self.max_joint_step,
        )

        # Send URScript servoj command
        cmd = (
            f"servoj({q_des.tolist()}, "
            f"t={self.servo_dt}, "
            f"lookahead_time={self.servo_lookahead}, "
            f"gain={self.servo_gain})\n"
        )
        self.conn.send(cmd.encode())

        # Gripper
        self._handle_gripper(float(action["gripper.pos"]))


    def _handle_gripper(self, value: float):
        value = float(np.clip(value, 0.0, 1.0))

        max_width = 160.0  # mm (RG6 fully open)
        min_width = 0.0

        target_width = max_width * (1.0 - value)
        current_width = self.rg6_gripper.get_rg_width()

        if abs(target_width - current_width) < 2.0:
            return

        self.rg6_gripper.set_rg_width(target_width)
