import time
import logging
import threading
import numpy as np
import cv2
from collections import deque
from queue import Queue
from pynput import keyboard

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("deploy")


# ─────────────────────────────────────────────
#  INTERVENTION THRESHOLDS — tune these
# ─────────────────────────────────────────────
STUCK_THRESHOLD       = 5.0    # seconds without TCP movement → stuck
GRIPPER_TIMEOUT       = 4.0    # seconds gripper unchanged during grasp attempt → stuck
FORCE_THRESHOLD       = 25.0   # Newtons — unexpected contact
ACTION_JUMP_THRESHOLD = 0.08   # max allowed action delta between steps
FRAME_DIFF_THRESHOLD  = 15.0   # pixel diff mean — scene changed unexpectedly
ACTION_HISTORY_LEN    = 20     # steps to track for jerk detection
FPS                   = 30


# ═══════════════════════════════════════════════════════
#  INTERVENTION MONITOR
# ═══════════════════════════════════════════════════════
class InterventionMonitor:
    def __init__(self, robot):
        self.robot = robot

        # Stuck detection
        self._last_tcp_pos    = None
        self._last_move_time  = time.time()

        # Gripper state tracking
        self._last_gripper_state  = None
        self._gripper_changed_time = time.time()
        self._waiting_for_gripper  = False

        # Action jerk detection
        self._action_history = deque(maxlen=ACTION_HISTORY_LEN)
        self._prev_action    = None

        # Frame diff detection
        self._prev_frame     = None

        # Stats for logging
        self.intervention_counts = {
            "human_requested": 0,
            "stuck":           0,
            "gripper_timeout": 0,
            "action_jerk":     0,
            "frame_anomaly":   0,
            "force_anomaly":   0,
        }

    def reset(self):
        """Call after human hands back to VLA — resets all timers."""
        self._last_move_time       = time.time()
        self._gripper_changed_time = time.time()
        self._waiting_for_gripper  = False
        self._prev_action          = None
        self._action_history.clear()
        self._prev_frame           = None
        logger.info("[MONITOR] Timers reset — VLA resuming")

    def check(self, obs: dict, action: dict, keys_queue: Queue) -> tuple[bool, str]:
        """
        Returns (should_intervene, reason).
        Call every step during autonomous operation.
        """

        # ── 1. Human on demand ────────────────────────────
        # Press 'h' anytime to take over
        if not keys_queue.empty():
            key = keys_queue.queue[0]
            if key == "h":
                keys_queue.get()
                self.intervention_counts["human_requested"] += 1
                return True, "human_requested"

        # ── 2. TCP stuck detection ─────────────────────────
        tcp = np.array([obs["tcp_pos.x"], obs["tcp_pos.y"], obs["tcp_pos.z"]])
        if self._last_tcp_pos is not None:
            if np.linalg.norm(tcp - self._last_tcp_pos) > 0.002:  # moved >2mm
                self._last_move_time = time.time()
        self._last_tcp_pos = tcp
        if time.time() - self._last_move_time > STUCK_THRESHOLD:
            self.intervention_counts["stuck"] += 1
            return True, "stuck"

        # ── 3. Gripper hasn't changed state ───────────────
        # Only trigger if VLA has been commanding gripper close for a while
        # and it hasn't actually closed (block not grasped)
        gripper_cmd = float(action.get("gripper.pos", 0.0))
        if gripper_cmd > 0.5:   # VLA wants to close
            self._waiting_for_gripper = True
        if self._waiting_for_gripper:
            current_gripper = obs.get("gripper.pos", 160.0)
            if self._last_gripper_state is not None:
                delta = abs(current_gripper - self._last_gripper_state)
                if delta > 2.0:   # gripper actually moved
                    self._gripper_changed_time = time.time()
                    self._waiting_for_gripper = False
            self._last_gripper_state = current_gripper
            if time.time() - self._gripper_changed_time > GRIPPER_TIMEOUT:
                self.intervention_counts["gripper_timeout"] += 1
                return True, "gripper_timeout"

        # ── 4. Action jerk detection ───────────────────────
        action_vec = np.array([action.get(f"joint.q{i}", 0.0) for i in range(6)])
        if self._prev_action is not None:
            jump = np.linalg.norm(action_vec - self._prev_action)
            if jump > ACTION_JUMP_THRESHOLD:
                self.intervention_counts["action_jerk"] += 1
                return True, "action_jerk"
        self._prev_action = action_vec
        self._action_history.append(action_vec)

        # ── 5. Frame diff — unexpected scene change ────────
        frame = obs.get("scene")
        if frame is not None and self._prev_frame is not None:
            gray_curr = cv2.cvtColor(frame,            cv2.COLOR_BGR2GRAY).astype(np.float32)
            gray_prev = cv2.cvtColor(self._prev_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            diff = np.abs(gray_curr - gray_prev).mean()
            if diff > FRAME_DIFF_THRESHOLD:
                self.intervention_counts["frame_anomaly"] += 1
                return True, "frame_anomaly"
        self._prev_frame = frame.copy() if frame is not None else None

        # ── 6. Force anomaly ───────────────────────────────
        try:
            wrench = self.robot.rtde_r.getActualTCPForce()
            if np.linalg.norm(wrench[:3]) > FORCE_THRESHOLD:
                self.intervention_counts["force_anomaly"] += 1
                return True, "force_anomaly"
        except Exception:
            pass

        return False, ""

    def print_stats(self):
        logger.info("─── Intervention Statistics ───")
        total = sum(self.intervention_counts.values())
        for reason, count in self.intervention_counts.items():
            logger.info(f"  {reason:<20}: {count}")
        logger.info(f"  {'TOTAL':<20}: {total}")


# ═══════════════════════════════════════════════════════
#  KEYBOARD LISTENER (shared with teleop)
# ═══════════════════════════════════════════════════════
class KeyListener:
    def __init__(self):
        self.queue = Queue()

    def start(self):
        def on_press(key):
            try:
                if key.char in ["h", "a", "q", "g"]:
                    self.queue.put(key.char)
            except AttributeError:
                if key == keyboard.Key.esc:
                    self.queue.put("esc")
        listener = keyboard.Listener(on_press=on_press)
        listener.daemon = True
        listener.start()


# ═══════════════════════════════════════════════════════
#  MAIN DEPLOY LOOP
# ═══════════════════════════════════════════════════════
def main():
    import ur5e
    import teleop
    import config_ur5e
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # ── Setup ──────────────────────────────────────────
    robot = ur5e.UR5e(
        config_ur5e.UR5eConfig(
            cameras={
                "scene": OpenCVCameraConfig(
                    index_or_path=0, width=432, height=240, fps=FPS
                ),
            },
        )
    )
    teleop_device  = teleop.UR5eTeleop(robot=robot)
    keys           = KeyListener()
    monitor        = InterventionMonitor(robot)

    # Dataset to log everything — interventions included
    dataset = LeRobotDataset.create(
        repo_id="dkhanh/ur5e_deploy_corrections",
        fps=FPS,
        features={
            "observation.state": {
                "dtype": "float32", "shape": (13,),
                "names": [
                    "tcp_pos.x", "tcp_pos.y", "tcp_pos.z",
                    "tcp_pos.r", "tcp_pos.p", "tcp_pos.yaw",
                    "joint.q0",  "joint.q1",  "joint.q2",
                    "joint.q3",  "joint.q4",  "joint.q5",
                    "gripper.pos",
                ],
            },
            "observation.images.scene": {
                "dtype": "video", "shape": (240, 432, 3),
                "names": ["height", "width", "channel"],
            },
            "action": {
                "dtype": "float32", "shape": (7,),
                "names": [
                    "joint.q0", "joint.q1", "joint.q2",
                    "joint.q3", "joint.q4", "joint.q5",
                    "gripper.pos",
                ],
            },
        },
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=1,
    )

    def build_frame(obs, action, task):
        state = np.array([
            obs["tcp_pos.x"], obs["tcp_pos.y"], obs["tcp_pos.z"],
            obs["tcp_pos.r"], obs["tcp_pos.p"], obs["tcp_pos.yaw"],
            obs["joint.q0"],  obs["joint.q1"],  obs["joint.q2"],
            obs["joint.q3"],  obs["joint.q4"],  obs["joint.q5"],
            obs["gripper.pos"],
        ], dtype=np.float32)
        action_vec = np.array([
            action.get("joint.q0", 0), action.get("joint.q1", 0),
            action.get("joint.q2", 0), action.get("joint.q3", 0),
            action.get("joint.q4", 0), action.get("joint.q5", 0),
            action.get("gripper.pos", 0),
        ], dtype=np.float32)
        return {
            "observation.state":        state,
            "observation.images.scene": obs["scene"],
            "action":                   action_vec,
            "task":                     task,
        }

    # ── Load your trained VLA here ─────────────────────
    # vla = load_your_vla_model()

    dt   = 1.0 / FPS
    mode = "auto"   # "auto" or "human"

    logger.info("Deploy started.")
    logger.info("  'h' = human take over")
    logger.info("  'a' = resume VLA autonomy")
    logger.info("  'g' = toggle gripper (human mode)")
    logger.info("  'q' = quit")

    try:
        robot.connect()
        teleop_device.connect()
        keys.start()

        episode_idx = 0
        running     = True

        while running:
            loop_start = time.perf_counter()

            # ── Quit check ─────────────────────────────
            if not keys.queue.empty():
                key = keys.queue.queue[0]
                if key in ["q", "esc"]:
                    keys.queue.get()
                    break

            obs = robot.get_observation()

            # ══════════════════════════════════════════
            #  AUTO MODE — VLA in control
            # ══════════════════════════════════════════
            if mode == "auto":
                # ── Get VLA action ─────────────────────
                # action = vla.predict(obs)  ← plug your VLA here
                action = {"joint.q0": 0.0, "joint.q1": 0.0, "joint.q2": 0.0,
                          "joint.q3": 0.0, "joint.q4": 0.0, "joint.q5": 0.0,
                          "gripper.pos": 0.0}  # placeholder

                # ── Check for intervention ─────────────
                should_intervene, reason = monitor.check(obs, action, keys.queue)

                if should_intervene:
                    logger.warning(f"[INTERVENTION] reason={reason} → human taking over")
                    print(f"\n{'='*50}")
                    print(f"  ⚠  INTERVENTION: {reason}")
                    print(f"  SpaceMouse active — press 'a' to resume VLA")
                    print(f"{'='*50}\n")
                    mode = "human"

                    # Save current episode before switching
                    dataset.save_episode()
                    episode_idx += 1

                else:
                    robot.send_action(action)
                    dataset.add_frame(build_frame(obs, action, "autonomous"))

            # ══════════════════════════════════════════
            #  HUMAN MODE — operator in control
            # ══════════════════════════════════════════
            elif mode == "human":
                action = teleop_device.get_action()
                robot.send_action(action)

                # Log human corrections — valuable training data
                dataset.add_frame(build_frame(obs, action, "human_correction"))

                # ── Resume VLA when human presses 'a' ─
                if not keys.queue.empty() and keys.queue.queue[0] == "a":
                    keys.queue.get()
                    logger.info("[HANDOVER] Human finished — VLA resuming")
                    print(f"\n{'='*50}")
                    print(f"  ✓  VLA resuming autonomy")
                    print(f"{'='*50}\n")

                    # Save human correction episode
                    dataset.save_episode()
                    episode_idx += 1

                    # Reset monitor timers — fresh start for VLA
                    monitor.reset()
                    mode = "auto"

            # ── Loop timing ────────────────────────────
            elapsed    = time.perf_counter() - loop_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                logger.warning(f"Loop slow: {1/elapsed:.1f} Hz")

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)

    finally:
        logger.info("Shutting down...")
        monitor.print_stats()

        # Save any unsaved episode
        try:
            dataset.save_episode()
            dataset.finalize()
        except Exception:
            pass

        # Stop robot
        stop = {f"joint.q{i}": float(robot.rtde_r.getActualQ()[i]) for i in range(6)}
        stop["gripper.pos"] = 0.0
        robot.send_action(stop)
        time.sleep(0.1)

        robot.disconnect()
        teleop_device.disconnect()
        logger.info("Done.")


if __name__ == "__main__":
    main()