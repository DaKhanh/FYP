import time
import logging
import numpy as np
from pathlib import Path

from autolab_core import RigidTransform
from autolab_core.transformations import euler_matrix

from UR5E.ur5e import UR5ERobot, SpaceMouseRobotController
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from UR5E import ur5e_cfg, ur5e  # your config files

# -------------------- CONFIG --------------------
DATASET_PATH = Path("ur5e_dataset")        # Folder to save dataset
NUM_EPISODES = 5
EPISODE_TIME_S = 60
RESET_TIME_S = 10
FPS = 10
TASK_NAME = "UR5e Teleop with SpaceMouse"

TRANSLATION_SCALE = 0.02  # SpaceMouse units -> meters
ROTATION_SCALE = 1 / 15   # SpaceMouse units -> radians

# -------------------- SETUP --------------------
logging.basicConfig(level=logging.INFO)

# Initialize robot with cameras
robot = ur5e.UR5e(
    ur5e_cfg.UR5eConfig(
        cameras={
            "wrist": RealSenseCameraConfig(serial_number_or_name="845112070856", width=240, height=240, fps=FPS),
            "scene": OpenCVCameraConfig(serial_number_or_name="CMSXJ22A", width=240, height=240, fps=FPS),
        }
    )
)

controller = SpaceMouseRobotController()

dataset = LeRobotDataset.create(
    repo_id=str(DATASET_PATH),
    fps=FPS,
    root=DATASET_PATH,
    robot_type="ur5e",
    features={**robot.observation_features, **robot.action_features},
    use_videos=True,  # record images/video now
)

robot.connect()
robot.open_gripper()

# -------------------- RECORD LOOP --------------------
for ep in range(NUM_EPISODES):
    logging.info(f"Starting episode {ep+1}/{NUM_EPISODES}")
    start_time = time.perf_counter()
    timestamp = 0

    while timestamp < EPISODE_TIME_S:
        loop_start = time.perf_counter()

        # --- Get controller input ---
        ctrl_state = controller.current_action
        translation = np.array(ctrl_state[:3]) * TRANSLATION_SCALE
        translation[0] *= -1  # mirror X
        translation[2] *= -1  # mirror Z
        rotation = np.array(ctrl_state[3:]) * ROTATION_SCALE
        rotation[0] = 0  # lock rotation X
        rotation[1] = 0  # lock rotation Y
        gripper_action = controller.gripper_state

        # --- Compute target pose ---
        delta_pose = RigidTransform(
            translation=translation,
            rotation=euler_matrix(*rotation, axes="ryxz")[:3, :3],
            from_frame="tcp",
            to_frame="tcp",
        )
        current_pose = robot.get_pose()
        current_pose.from_frame = "tcp"
        new_pose = current_pose * delta_pose

        # --- Send action to robot ---
        robot.servo_pose(new_pose, time=0.01, lookahead_time=0.2, gain=100)
        robot._handle_gripper(gripper_action)

        # --- Save observation + action + images ---
        obs = robot.get_observation()
        action = {
            "tcp.x": translation[0],
            "tcp.y": translation[1],
            "tcp.z": translation[2],
            "tcp.rx": rotation[0],
            "tcp.ry": rotation[1],
            "tcp.rz": rotation[2],
            "gripper.pos": gripper_action,
        }
        dataset.add_frame({**obs, **action, "task": TASK_NAME})

        # --- Maintain FPS ---
        dt = time.perf_counter() - loop_start
        sleep_time = max(1 / FPS - dt, 0)
        time.sleep(sleep_time)
        timestamp = time.perf_counter() - start_time

    # --- Reset robot for next episode ---
    logging.info("Episode finished, resetting robot...")
    time.sleep(RESET_TIME_S)
    dataset.save_episode()

# -------------------- CLEANUP --------------------
dataset.finalize()
robot.disconnect()
logging.info("Dataset recording finished!")