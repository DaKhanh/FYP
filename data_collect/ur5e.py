import socket
import logging
import numpy as np
import time
from ur_rtde import rtde_receive
from ur_rtde import dashboard_client
import onRobot.gripper as gripper
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots.robot import Robot
import config_ur5e as ur5e_cfg

logger = logging.getLogger(__name__)

class UR5e(Robot):
    config_class = ur5e_cfg.UR5eConfig
    name = "ur5e"

    def __init__(self, config: ur5e_cfg.UR5eConfig):
        super().__init__(config)

        self.config = config
        self.robot_ip = config.robot_ip
        self.robot_port = config.robot_port

        # RTDE (receive only)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(self.robot_ip)
        self.dash_c = dashboard_client.DashboardClient(self.robot_ip)

        # Gripper
        self.rg6_gripper = gripper.RG2()
        self._last_gripper_target = 160.0

        # Servo socket
        self.conn = None
        self.control_freq = config.control_freq
        self.servo_dt = 1.0 / self.control_freq
        # Cameras
        self.cameras = make_cameras_from_configs(config.cameras)

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
        # Updated to include tcp_pos and gripper
        return {
            "tcp_pos.x": float,
            "tcp_pos.y": float,
            "tcp_pos.z": float,
            "tcp_pos.r": float,
            "tcp_pos.p": float,
            "tcp_pos.yaw": float,
            "gripper.pos": float,
            **self._cameras_ft,  # Adding camera features
        }

    @property
    def action_features(self) -> dict[str, type]:
        # This must match what is returned by get_action()
        return {
            "vx": float, "vy": float, "vz": float,
            "gripper.pos": float,
        }

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {
            "vx": float, "vy": float, "vz": float,
            "gripper.pos": float,
        }

    @property
    def is_connected(self) -> bool:
        return (
            self.rtde_r.isConnected()
            and self.dash_c.isConnected()
            and all(cam.is_connected for cam in self.cameras.values())
        )

    def connect(self, calibrate=True):
        self.dash_c.connect()

        self.conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.conn.connect((self.robot_ip, self.robot_port))

        for cam in self.cameras.values():
            cam.connect()

        logger.info("UR5e connected")

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        logger.info("UR5e calibration skipped (assumed calibrated)")

    def configure(self) -> None:
        logger.info("UR5e configuration skipped")

    def disconnect(self) -> None:
        if self.conn is not None:
            self.conn.send(b"stopj(0.5)\n")
            time.sleep(0.1)
            self.conn.close()
            self.conn = None
        for cam in self.cameras.values():
            cam.disconnect()

    def get_observation(self) -> dict[str, np.ndarray]:
        obs = {}

        # Get the current TCP pose
        tcp_pose = self.rtde_r.getActualTCPPose()
        obs["tcp_pos.x"] = np.float32(tcp_pose[0])
        obs["tcp_pos.y"] = np.float32(tcp_pose[1])
        obs["tcp_pos.z"] = np.float32(tcp_pose[2])
        obs["tcp_pos.r"] = np.float32(tcp_pose[3])
        obs["tcp_pos.p"] = np.float32(tcp_pose[4])
        obs["tcp_pos.yaw"] = np.float32(tcp_pose[5])

        # Gripper position
        obs["gripper.pos"] = np.float32(self._last_gripper_target)

        # Cameras
        for cam_name, cam in self.cameras.items():
            obs[cam_name] = cam.async_read()

        return obs

    def send_action(self, action):
        action_dict = {
            "vx":  float(action.get("vx", 0)),
            "vy":  float(action.get("vy", 0)),
            "vz":  float(action.get("vz", 0)),
            "gripper.pos": float(action.get("gripper.pos", 0)),
        }

        self._write_to_motors(action_dict)
        return action_dict

    def _write_to_motors(self, action, a: float = 0.5, t: float = 0.033): 
        if self.conn is None:
            return

        # Extract velocities
        v = [action["vx"], action["vy"], action["vz"]] 

        cmd = f"speedl([{v[0]},{v[1]},{v[2]},0.0,0.0,0.0], a={a}, t={t})\n"
        
        self.conn.send(cmd.encode())
        self._handle_gripper(float(action["gripper.pos"]))

    def _handle_gripper(self, value: float):
        value = float(np.clip(value, 0.0, 1.0))
        target_width = 160.0 * (1.0 - value)
        
        # Only send if target changed AND gripper has had time to move
        if not hasattr(self, '_last_gripper_target'):
            self._last_gripper_target = None
            
        if target_width != self._last_gripper_target:
            self._last_gripper_target = target_width
            print(f"[GRIPPER] Sending rg_grip({target_width:.1f})")
            self.rg6_gripper.rg_grip(target_width)

