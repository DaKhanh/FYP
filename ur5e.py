import socket
import time
import logging
import numpy as np
from ur_rtde import rtde_receive, dashboard_client
import onRobot.gripper as gripper
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots.robot import Robot
import config_ur5e as ur5e_cfg
import ik_ur5e
logger = logging.getLogger(__name__)


class UR5e(Robot):
    config_class = ur5e_cfg.UR5eConfig
    name = "ur5e"

    def __init__(self, config: ur5e_cfg.UR5eConfig):
        super().__init__(config)
        self.config     = config
        self.robot_ip   = config.robot_ip
        self.robot_port = config.robot_port
        self.ik = ik_ur5e.IK_UR5e()
        # RTDE receive only
        self.rtde_r = rtde_receive.RTDEReceiveInterface(self.robot_ip)
        self.dash_c = dashboard_client.DashboardClient(self.robot_ip)

        # Gripper
        self.rg6_gripper          = gripper.RG2()
        self._last_gripper_target = 160.0   # start open (mm)

        # Socket for URScript commands
        self.conn = None

        # Cameras
        self.cameras = make_cameras_from_configs(config.cameras)

        # servoj params — tune for smoothness
        self._servo_t          = 0.033   # must match 1/FPS
        self._servo_lookahead  = 0.1    # 0.03–0.2s, higher = smoother
        self._servo_gain       = 150     # 100–2000, lower = softer

        # Safety: max TCP delta per step
        self.max_step_m = 0.03         # 5mm per step max

    @property
    def _cameras_ft(self) -> dict:
        return {
            cam: (
                self.config.cameras[cam].height,
                self.config.cameras[cam].width,
                3,
            )
            for cam in self.cameras
        }

    @property
    def observation_features(self) -> dict:
        return {
            "tcp_pos.x": float, "tcp_pos.y": float, "tcp_pos.z": float,
            "tcp_pos.r": float, "tcp_pos.p": float, "tcp_pos.yaw": float,
            "joint.q0": float, "joint.q1": float, "joint.q2": float,
            "joint.q3": float, "joint.q4": float, "joint.q5": float,
            "gripper.pos": float,
            **self._cameras_ft,
        }

    @property
    def action_features(self) -> dict:
        return {
            "joint.q0": float, "joint.q1": float, "joint.q2": float,
            "joint.q3": float, "joint.q4": float, "joint.q5": float,
            "gripper.pos": float,
        }

    @property
    def is_connected(self) -> bool:
        return self.rtde_r.isConnected() and self.conn is not None

    def connect(self, calibrate=True):
        self.dash_c.connect()
        self.conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.conn.connect((self.robot_ip, self.robot_port))
        print(f"[INFO] Socket connected to {self.robot_ip}:{self.robot_port}")
        for cam in self.cameras.values():
            cam.connect()
        logger.info("UR5e connected")

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self)  -> None: pass
    def configure(self)        : pass

    def disconnect(self) -> None:
        if self.conn is not None:
            try:
                self.conn.send(b"stopj(0.5)\n")
                time.sleep(0.1)
            except Exception:
                pass
            self.conn.close()
            self.conn = None
        for cam in self.cameras.values():
            cam.disconnect()
        logger.info("UR5e disconnected")

    def get_observation(self) -> dict:
        obs = {}

        tcp = self.rtde_r.getActualTCPPose()
        obs["tcp_pos.x"]   = np.float32(tcp[0])
        obs["tcp_pos.y"]   = np.float32(tcp[1])
        obs["tcp_pos.z"]   = np.float32(tcp[2])
        obs["tcp_pos.r"]   = np.float32(tcp[3])
        obs["tcp_pos.p"]   = np.float32(tcp[4])
        obs["tcp_pos.yaw"] = np.float32(tcp[5])

        joints = self.rtde_r.getActualQ()
        for i, q in enumerate(joints):
            obs[f"joint.q{i}"] = np.float32(q)

        # Cached gripper — avoids slow serial read every frame
        obs["gripper.pos"] = np.float32(self._last_gripper_target)

        for cam_name, cam in self.cameras.items():
            obs[cam_name] = cam.async_read()

        return obs

    def send_action(self, action: dict) -> dict:
        if self.conn is None:
            return action  # return as-is so recording still works

        q = [float(action.get(f"joint.q{i}", 0.0)) for i in range(6)]
        cmd = (
            f"servoj([{q[0]:.6f},{q[1]:.6f},{q[2]:.6f},"
            f"{q[3]:.6f},{q[4]:.6f},{q[5]:.6f}],"
            f"t={self._servo_t},"
            f"lookahead_time={self._servo_lookahead},"
            f"gain={self._servo_gain})\n"
        )
        self.conn.send(cmd.encode())
        self._handle_gripper(float(action.get("gripper.pos", 0.0)))
        return action 

    def _handle_gripper(self, value: float):
        value        = float(np.clip(value, 0.0, 1.0))
        target_width = 160.0 * (1.0 - value)
        if target_width != self._last_gripper_target:
            self._last_gripper_target = target_width
            print(f"[GRIPPER] → {target_width:.1f}mm")
            self.rg6_gripper.rg_grip(target_width, target_force=30)

    def go_home(self, home_joints=None):
        """Move robot to home joint configuration using movej."""
        if self.conn is None:
            return
        if home_joints is None:
            home_joints = [0.8417584896087646, -2.4198800526061, 2.0685880819903772,
                       -1.2207814317992707, -1.5687277952777308, -0.7282212416278284]
        q = home_joints
        cmd = (
            f"movej([{q[0]:.6f},{q[1]:.6f},{q[2]:.6f},"
            f"{q[3]:.6f},{q[4]:.6f},{q[5]:.6f}],"
            f"a=0.5, v=0.3)\n"
        )
        self.conn.send(cmd.encode())
        time.sleep(4.0)  # wait for movej to complete
        print("[INFO] Robot returned to home position")

# ssh E220025@172.21.89.130