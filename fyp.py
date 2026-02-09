import numpy as np
import torch
import cv2
import config_ur5e as ur5e_cfg
import ur5e as ur5e
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

TASK_DESCRIPTION = "pick the red cube and place it on the table"
REPO_ID = "dkhanh/xvla-pick-red-cube"
DATASET = "dkhanh/pick-red-cube"


def pack_state_from_robot_obs(obs: dict) -> np.ndarray:
    state = []
    for i in range(1, 7):
        state.append(float(obs.get(f"joint_{i}.pos", 0.0)))
    state.append(float(obs.get("gripper.pos", 0.0)))
    return np.asarray(state, dtype=np.float32)

def prepare_obs_dict(obs: dict) -> dict:
    images = {}
    def add_img(key, size):
        if key in robot.cameras:
            img = obs[key]
            img = cv2.resize(img, size)
            images[key] = np.clip(img, 0, 255).astype(np.uint8)

    add_img("wrist", (240, 240))
    add_img("scene", (240, 240))

    obs_dict = {
        "observation.images.image": images.get("wrist"),
        "observation.images.image2": images.get("scene"),
        "observation.state": pack_state_from_robot_obs(obs),
        "task": TASK_DESCRIPTION,
    }

    return obs_dict

print("Loading XVLA policy...")
policy = XVLAPolicy.from_pretrained(REPO_ID).cuda()
policy.eval()
print("Policy ready")

# Load dataset stats for pre/post processors
train_dataset = LeRobotDataset(repo_id=DATASET)
dataset_stats = train_dataset.meta.stats

preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=policy.config,
    pretrained_path=REPO_ID,
    dataset_stats=dataset_stats,
)
print("[Server] Pre/Post processors ready")

robot = ur5e.UR5e(
    ur5e_cfg.UR5eConfig(
        cameras={
            "wrist": RealSenseCameraConfig(serial_number_or_name="845112070856", width=240, height=240, fps=10),
            "scene": OpenCVCameraConfig(serial_number_or_name="CMSXJ22A", width=240, height=240, fps=10),        },
    )
)
robot.connect()
print("[Robot] Connected")
print(robot.cameras)


try:

    while True:

        # 1. Get robot observation
        obs = robot.get_observations()
        obs_dict = prepare_obs_dict(obs)

        # 2. Send to remote policy, receive action
        observation_preprocessed = preprocessor(obs_dict)
        action_chunk = policy.predict_action_chunk(obs_dict)
        action_postprocessed = postprocessor(action_chunk)
        action_vec = action_postprocessed[0]
        for a in action_vec:
            action_tensor = torch.as_tensor(a, dtype=torch.float32)
            robot.send_action(action_tensor)
            print("Sent action to robot:", action_tensor.numpy())


finally:
    print("Disconnecting robot...")
    robot.disconnect()

