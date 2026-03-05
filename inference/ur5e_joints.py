import socket
import time
import logging
import numpy as np
from typing import List

from ur_rtde import rtde_receive
from ur_rtde import dashboard_client
import onRobot.gripper as gripper
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots.robot import Robot
from .config_ur5e import UR5eConfig
from autolab_core import RigidTransform
from autolab_core.transformations import euler_matrix

logger = logging.getLogger(__name__)


def RT2UR(rt: RigidTransform):
    """
    Convert RigidTransform to UR pose: [x, y, z, rx, ry, rz]
    """
    return rt.translation.tolist() + rt.axis_angle.tolist()


def UR2RT(pose: List[float]) -> RigidTransform:
    """
    Convert UR pose [x,y,z,rx,ry,rz] to RigidTransform
    """
    rot = RigidTransform.rotation_from_axis_angle(pose[-3:])
    return RigidTransform(translation=pose[:3], rotation=rot)


class UR5e(Robot):
    config_class = UR5eConfig
    name = "ur5e"

    def __init__(self, config: UR5eConfig):
        super().__init__(config)

        self.config = config
        self.robot_ip = config.robot_ip
        self.robot_port = config.robot_port

        # RTDE receive interface
        self.rtde_r = rtde_receive.RTDEReceiveInterface(self.robot_ip)
        self.dash_c = dashboard_client.DashboardClient(self.robot_ip)

        # Gripper
        self.rg6_gripper = gripper.RG6()

        # Servo socket
        self.conn = None

        # Servo parameters
        self.control_freq = config.control_freq
        self.servo_dt = 1.0 / self.control_freq
        self.servo_lookahead = config.servo_lookahead
        self.servo_gain = config.servo_gain

        # Safety
        self.max_tcp_step = 0.05  # max movement per step (m or rad)

        # Cameras
        self.cameras = make_cameras_from_configs(config.cameras)

    # -------------------- OBSERVATION / ACTION --------------------
    @property
    def _motors_ft(self) -> dict[str, type]:
        ft = {f"tcp.{axis}": float for axis in ["x", "y", "z", "rx", "ry", "rz"]}
        ft["gripper.pos"] = float
        return ft

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height,
                  self.config.cameras[cam].width,
                  3)
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

    # -------------------- CONNECTION --------------------
    def connect(self, calibrate: bool = True) -> None:
        self.dash_c.connect()
        self.conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.conn.connect((self.robot_ip, self.robot_port))
        for cam in self.cameras.values():
            cam.connect()
        logger.info("UR5e TCP control connected")

    def disconnect(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None
        for cam in self.cameras.values():
            cam.disconnect()
        logger.info("UR5e disconnected")

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        logger.info("UR5e calibration skipped (assumed calibrated)")

    def configure(self) -> None:
        logger.info("UR5e configuration skipped")

    # -------------------- TCP POSE READ / WRITE --------------------
    def get_pose(self) -> RigidTransform:
        """
        Return current TCP pose as RigidTransform
        """
        p = self.rtde_r.getActualTCPPose()  # [x, y, z, rx, ry, rz]
        return UR2RT(p)

    def servo_pose(self, target: RigidTransform, time: float = 0.002,
                   lookahead_time: float = 0.1, gain: float = 300):
        """
        Move robot to target RigidTransform pose
        """
        pos = RT2UR(target)
        print(f"Moving to >> Translation: {pos[:3]} | Rotation: {pos[3:]}")
        if self.conn:
            cmd = f"servoj({pos}, t={time}, lookahead_time={lookahead_time}, gain={gain})\n"
            self.conn.send(cmd.encode())

    # -------------------- ABSTRACT OVERRIDE --------------------
    def get_observation(self) -> dict[str, np.ndarray]:
        obs = {}
        tcp_pose = self.get_pose()
        obs["tcp.x"] = np.float32(tcp_pose.translation[0])
        obs["tcp.y"] = np.float32(tcp_pose.translation[1])
        obs["tcp.z"] = np.float32(tcp_pose.translation[2])
        obs["tcp.rx"] = np.float32(tcp_pose.axis_angle[0])
        obs["tcp.ry"] = np.float32(tcp_pose.axis_angle[1])
        obs["tcp.rz"] = np.float32(tcp_pose.axis_angle[2])

        obs["gripper.pos"] = np.float32(self.rg6_gripper.get_rg_width())

        for cam_name, cam in self.cameras.items():
            obs[cam_name] = cam.async_read()

        return obs

    def send_action(self, action_list):
        """
        action_list: [x, y, z, rx, ry, rz, gripper]
        """
        target_pose = RigidTransform(
            translation=np.array(action_list[:3]),
            rotation=euler_matrix(*action_list[3:], axes="ryxz")[:3, :3],
            from_frame="tcp",
            to_frame="tcp"
        )
        self.servo_pose(target_pose, time=self.servo_dt,
                        lookahead_time=self.servo_lookahead,
                        gain=self.servo_gain)

        self._handle_gripper(float(action_list[6]))

    # -------------------- GRIPPER --------------------
    def _handle_gripper(self, value: float):
        value = float(np.clip(value, 0.0, 1.0))
        max_width = 160.0  # mm (RG6 fully open)
        target_width = max_width * (1.0 - value)
        current_width = self.rg6_gripper.get_rg_width()
        if abs(target_width - current_width) < 2.0:
            return
        self.rg6_gripper.set_rg_width(target_width)