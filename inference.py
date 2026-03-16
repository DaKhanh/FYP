import numpy as np
import ur5e
import config_ur5e 
import socket
import time
import pickle
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

TASK_DESCRIPTION = "put the cylinder on top of the cube"
HOST = "10.97.26.171"  # Remote GPU IP
PORT = 5002
FPS=30

def obs_to_state(obs: dict) -> np.ndarray:
    return np.array([
        obs["tcp_pos.x"], obs["tcp_pos.y"], obs["tcp_pos.z"],
        obs["tcp_pos.r"], obs["tcp_pos.p"], obs["tcp_pos.yaw"],
        obs["joint.q0"],  obs["joint.q1"],  obs["joint.q2"],
        obs["joint.q3"],  obs["joint.q4"],  obs["joint.q5"],
        obs["gripper.pos"],
    ], dtype=np.float32)

def prepare_obs_dict(obs: dict) -> dict:
    images = {}
    
    def add_img(key):
        if key in obs:
            img = obs[key]  # HWC uint8
            img = np.clip(img, 0, 255).astype(np.uint8)
            img = np.transpose(img, (2, 0, 1))  # CHW
            img = np.expand_dims(img, 0)        # B, C, H, W
            images[key] = img

    add_img("scene")

    obs_dict = {
        "observation.images.image": images.get("scene"),
        "observation.state": obs_to_state(obs),
        "task": TASK_DESCRIPTION,
    }

    return obs_dict


robot = ur5e.UR5e(
        config_ur5e.UR5eConfig(
            cameras={
                "scene": OpenCVCameraConfig(index_or_path=0, width=432, height=240, fps=FPS),
            },
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
        for a in actions[0][:10]:
            action_dict = {f"joint.q{i}": float(a[i]) for i in range(6)}
            action_dict["gripper.pos"] = float(a[6])

            robot.send_action(action_dict)
            time.sleep(1/FPS)



