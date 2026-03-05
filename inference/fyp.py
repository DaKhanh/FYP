import numpy as np
import config_ur5e_joints as ur5e_cfg
import ur5e_joints 
import socket
import pickle
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

TASK_DESCRIPTION = "pick the cube and place it on the table"
REPO_ID = "dkhanh/xvla-pick-red-cube"
DATASET = "dkhanh/pick-red-cube"
HOST = "10.97.26.171"  # Remote GPU IP
PORT = 5001

def pack_state_from_robot_obs(obs: dict) -> np.ndarray:
    state = []
    for i in range(1, 7):
        state.append(float(obs.get(f"joint_{i}.pos", 0.0)))
    state.append(float(obs.get("gripper.pos", 0.0)))
    return np.asarray(state, dtype=np.float32)

def prepare_obs_dict(obs: dict) -> dict:
    images = {}
    
    def add_img(key):
        if key in obs:
            img = obs[key]  # HWC uint8
            img = np.clip(img, 0, 255).astype(np.uint8)
            img = np.transpose(img, (2, 0, 1))  # CHW
            img = np.expand_dims(img, 0)        # B, C, H, W
            images[key] = img

    add_img("wrist")
    add_img("scene")

    obs_dict = {
        "observation.images.image": images.get("wrist"),
        "observation.images.image2": images.get("scene"),
        "observation.state": pack_state_from_robot_obs(obs),
        "task": TASK_DESCRIPTION,
    }

    return obs_dict


robot = ur5e_joints.UR5e(
    ur5e_cfg.UR5eConfig(
        cameras={
            "wrist": OpenCVCameraConfig(index_or_path="/dev/video4", width=424, height=240, fps=15),
            "scene": OpenCVCameraConfig(index_or_path="/dev/video6", width=432, height=240, fps=15),        },
    )
)
robot.connect()
print("[Robot] Connected")
print(robot.cameras)


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    while True:
        # 1. Get observation
        obs = robot.get_observation()
        obs_dict = prepare_obs_dict(obs)

        # 2. Serialize and send
        data_bytes = pickle.dumps(obs_dict)
        s.sendall(len(data_bytes).to_bytes(8, "big") + data_bytes)

        # 3. Receive actions
        data_len_bytes = s.recv(8)
        data_len = int.from_bytes(data_len_bytes, "big")
        data = b""
        while len(data) < data_len:
            data += s.recv(data_len - len(data))

        actions = pickle.loads(data)

        # 4. Send actions to robot
        for a in actions[0]:
            robot.send_action(a)



