import logging
import time
import numpy as np
import ur5e
import teleop
import config_ur5e
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.utils import log_say

logging.basicConfig(level=logging.WARNING)

FPS = 30
REPO_ID = "dkhanh/ur5e_data_collect"
STATE_DIM = 7   
ACTION_DIM = 4  


class RecordConfig:
    def __init__(self):
        self.num_episodes = 1
        self.display = False
        self.task_description = "put the blocks to the box"
        self.episode_time_sec = 60
        self.reset_time_sec = 10
        self.push_to_hub = False


def build_dataset_features() -> dict:
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (STATE_DIM,),
            "names": ["tcp_pos.x", "tcp_pos.y", "tcp_pos.z",
                      "tcp_pos.r", "tcp_pos.p", "tcp_pos.yaw",
                      "gripper.pos"],
        },
        "observation.images.scene": {
            "dtype": "video",
            "shape": (240, 432, 3),
            "names": ["height", "width", "channel"],
        },
        "action": {
            "dtype": "float32",
            "shape": (ACTION_DIM,),
            "names": ["vx", "vy", "vz", "gripper.pos"],
        },
    }


def obs_to_state(obs: dict) -> np.ndarray:
    return np.array([
        obs["tcp_pos.x"], obs["tcp_pos.y"], obs["tcp_pos.z"],
        obs["tcp_pos.r"], obs["tcp_pos.p"], obs["tcp_pos.yaw"],
        obs["gripper.pos"],
    ], dtype=np.float32)


def action_to_vector(action: dict) -> np.ndarray:
    return np.array([
        action["vx"],  action["vy"],  action["vz"],
        action["gripper.pos"],
    ], dtype=np.float32)


def build_frame(obs: dict, action: dict, task: str) -> dict:
    return {
        "observation.state": obs_to_state(obs),
        "observation.images.scene": obs["scene"],
        "action": action_to_vector(action),
        "task": task,
    }


def main(record_cfg: RecordConfig):
    # Setup robot
    robot = ur5e.UR5e(
        config_ur5e.UR5eConfig(
            cameras={
                "scene": OpenCVCameraConfig(index_or_path=1, width=432, height=240, fps=FPS),
            },
        )
    )
    robot.connect()
    print("Robot connected")

    # Setup teleop
    teleop_device = teleop.UR5eTeleop(robot=robot)
    teleop_device.connect()

    # Create dataset with manually defined features
    dataset_features = build_dataset_features()
    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=FPS,
        features=dataset_features,
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=1,
    )

    # Keyboard events
    events = {
        "exit_early": False,
        "stop_recording": False,
        "rerecord_episode": False,
    }

    def check_keys():
        while not teleop_device.misc_keys_queue.empty():
            key = teleop_device.misc_keys_queue.get()
            if key in ["q", "esc"]:
                events["stop_recording"] = True
            elif key == "s":
                events["exit_early"] = True
            elif key == "r":
                events["rerecord_episode"] = True
                events["exit_early"] = True

    dt = 1.0 / FPS
    episode_idx = 0

    while episode_idx < record_cfg.num_episodes and not events["stop_recording"]:
        log_say(f"Recording episode {episode_idx + 1} of {record_cfg.num_episodes}", play_sounds=False)
        print(f"\n=== Episode {episode_idx + 1}/{record_cfg.num_episodes} ===")
        print("Move SpaceMouse to record. Press 's' to save, 'r' to re-record, 'q' to quit.")

        # Clear stale keys
        while not teleop_device.misc_keys_queue.empty():
            teleop_device.misc_keys_queue.get()
        events["exit_early"] = False
        events["rerecord_episode"] = False

        # --- Record episode ---
        start_t = time.perf_counter()
        timestamp = 0.0

        while timestamp < record_cfg.episode_time_sec:
            loop_start = time.perf_counter()
            check_keys()

            if events["exit_early"] or events["stop_recording"]:
                events["exit_early"] = False
                break

            obs = robot.get_observation()
            action = teleop_device.get_action()
            robot.send_action(action)

            frame = build_frame(obs, action, record_cfg.task_description)
            dataset.add_frame(frame)

            elapsed = time.perf_counter() - loop_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                logging.warning(f"Loop running slow: {1/elapsed:.1f} Hz vs target {FPS} Hz")

            timestamp = time.perf_counter() - start_t

        # --- Handle re-record ---
        if events["rerecord_episode"]:
            print("Re-recording episode...")
            events["rerecord_episode"] = False
            dataset.clear_episode_buffer()
            continue

        if events["stop_recording"]:
            dataset.clear_episode_buffer()
            break

        # --- Save episode ---
        dataset.save_episode()
        print(f"Episode {episode_idx + 1} saved.")
        episode_idx += 1

        # --- Reset phase (skip after last episode) ---
        if episode_idx < record_cfg.num_episodes and not events["stop_recording"]:
            print(f"Reset environment. You have {record_cfg.reset_time_sec}s...")
            reset_start = time.perf_counter()
            while time.perf_counter() - reset_start < record_cfg.reset_time_sec:
                check_keys()
                if events["stop_recording"]:
                    break
                obs = robot.get_observation()
                action = teleop_device.get_action()
                robot.send_action(action)
                time.sleep(dt)

    # --- Finalize ---
    print("Finalizing dataset...")
    dataset.finalize()

    if record_cfg.push_to_hub:
        dataset.push_to_hub()
    stop = {"vx": 0.0, "vy": 0.0, "vz": 0.0, "gripper.pos": 0.0}
    robot.send_action(stop)
    time.sleep(0.1)
    robot.disconnect()
    teleop_device.disconnect()
    print("Done.")

if __name__ == "__main__":
    record_cfg = RecordConfig()
    main(record_cfg)