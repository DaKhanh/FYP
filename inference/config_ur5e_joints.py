from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("ur5e")
@dataclass
class UR5eConfig(RobotConfig):
    robot_ip: str = "192.168.1.30"
    robot_port: int = 30002
    servo_lookahead: float = 0.1
    servo_gain: int = 300
    max_joint_step: float = 0.05  
    control_freq: int = 20

    # cameras (shared between both arms)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)