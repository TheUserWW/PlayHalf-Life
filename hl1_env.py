from pathlib import Path
from collections import deque
import ctypes
import json
import re
import tempfile
import threading
import time

import cv2
import gymnasium as gym
import mss
import numpy as np
import pydirectinput
from gymnasium import spaces

from audio_feedback import FootstepAudioDetector
from depth_perception import RelativeDepthEstimator


class HalfLifeEnv(gym.Env):
    """Screen-based Gymnasium environment with ordered visual checkpoints."""

    metadata = {"render_modes": ["human"]}
    IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
    # Movement-plus-jump choices. Opposing directions are omitted because they cancel.
    ACTION_KEYS = {
        0: (),
        1: ("w",),
        4: ("space",),
        5: ("w", "ctrl", "space"),
        6: ("w", "space"),
        7: ("a", "space"),
        8: ("s", "space"),
        9: ("d", "space"),
        10: ("a", "w", "space"),
        11: ("w", "d", "space"),
        12: ("a", "s", "space"),
        13: ("s", "d", "space"),
        14: ("s",),
        15: ("a",),
        16: ("d",),
    }
    JUMP_ACTIONS = frozenset(range(4, 14))
    MOVEMENT_ACTIONS = frozenset({1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16})
    BASE_ACTION_COUNT = 17
    TERMINATION_CHECKPOINT_NAME = "d"
    MAX_EPISODE_STEPS = 500
    SUCCESS_TIME_BONUS = 1.0
    INCOMPLETE_SCORE_FACTOR = 0.25
    INCOMPLETE_EPISODE_PENALTY = 1.0
    # Config binds that can interrupt training, open menus, or alter save state.
    BLOCKED_BIND_KEYS = frozenset({"ESCAPE", "`", "~", "F6", "F7", "F10", "PAUSE"})
    BLOCKED_BIND_COMMANDS = (
        "toggleconsole",
        "cancelselect",
        "quit",
        "pause",
        "save",
        "load",
        "messagemode",
        "messagemode2",
        "say",
        "voicerecord",
        "impulse 201",
    )
    KEY_ALIASES = {
        "ESCAPE": "esc",
        "UPARROW": "up",
        "DOWNARROW": "down",
        "LEFTARROW": "left",
        "RIGHTARROW": "right",
        "INS": "insert",
        "PGDN": "pagedown",
        "PGUP": "pageup",
        "+": ("shift", "="),
        "~": ("shift", "`"),
    }
    DIRECTION_BIND_KEYS = frozenset(
        {"UPARROW", "DOWNARROW", "LEFTARROW", "RIGHTARROW"}
    )

    def __init__(
        self,
        monitor=None,
        action_duration=0.1,
        turn_action_duration=0.35,
        jump_action_duration=0.35,
        crouch_jump_duration=0.35,
        checkpoint_threshold=0.60,
        checkpoint_confirm_frames=3,
        stuck_similarity_threshold=0.90,
        stuck_min_movement_seconds=1.5,
        stuck_confirm_steps=3,
        stuck_penalty=1.0,
        footstep_match_threshold=0.82,
        footstep_reward=0.002,
        config_path=None,
        checkpoint_root=None,
    ):
        super().__init__()
        # Keep multi-key actions near-simultaneous; action timing is controlled below.
        pydirectinput.PAUSE = 0
        config_path = Path(config_path) if config_path else Path(__file__).resolve().parent / "config.cfg"
        self.config_bindings = self._load_config_bindings(config_path)
        self.action_space = spaces.Discrete(self.BASE_ACTION_COUNT + len(self.config_bindings))
        self.monitor = monitor or {"top": 0, "left": 0, "width": 640, "height": 480}
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            # Four temporal grayscale frames, each paired with estimated relative depth.
            shape=(self.monitor["height"], self.monitor["width"], 8),
            dtype=np.uint8,
        )
        # Keep enough frames to select one view from each second of the last 3 seconds.
        self._frame_history = deque(maxlen=128)
        self._frame_history_lock = threading.Lock()
        self._capture_stop = threading.Event()
        self.action_duration = action_duration
        self.turn_action_duration = turn_action_duration
        self.jump_action_duration = jump_action_duration
        self.crouch_jump_duration = crouch_jump_duration
        self.checkpoint_threshold = checkpoint_threshold
        self.checkpoint_confirm_frames = checkpoint_confirm_frames
        self.stuck_similarity_threshold = stuck_similarity_threshold
        self.stuck_min_movement_seconds = stuck_min_movement_seconds
        self.stuck_confirm_steps = stuck_confirm_steps
        self.stuck_penalty = stuck_penalty
        self.footstep_reward = footstep_reward
        self._recent_actions = deque(maxlen=128)
        self._stuck_hits = 0
        self._is_stuck = False
        self._held_keys = set()
        self._held_mouse_buttons = set()
        self.footstep_detector = FootstepAudioDetector(
            Path(__file__).resolve().parent / "sounds" / "Movement",
            threshold=footstep_match_threshold,
        )
        print(self.footstep_detector.status)
        self.depth_estimator = RelativeDepthEstimator()
        self.sct = mss.mss()
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name="hl1-screen-capture",
            daemon=True,
        )
        self._capture_thread.start()

        root = Path(checkpoint_root) if checkpoint_root else Path(__file__).resolve().parent / "rewards"
        self.checkpoint_folders = self._discover_checkpoint_folders(root)
        self.checkpoint_images = self._load_checkpoint_images(self.checkpoint_folders)
        self.termination_checkpoint_count = next(
            (
                index + 1
                for index, folder in enumerate(self.checkpoint_folders)
                if folder.name.casefold() == self.TERMINATION_CHECKPOINT_NAME
            ),
            None,
        )
        if self.termination_checkpoint_count is None:
            raise ValueError(
                f"Required termination checkpoint folder not found: "
                f"{self.TERMINATION_CHECKPOINT_NAME}"
            )
        self.checkpoint_index = 0
        self._claimed_checkpoints = set()
        self._previous_similarity = 0.0
        self._checkpoint_hits = [0] * len(self.checkpoint_images)
        self._episode_steps = 0
        self._episode_id = str(time.time_ns())
        self._episode_actions = []
        self._last_observation = None
        self._behavior_root = Path(__file__).resolve().parent / "successful_behaviors"
        self._behavior_temp = None
        self._episode_capture_dir = None
        self._start_behavior_capture()

    def _discover_checkpoint_folders(self, root):
        if not root.is_dir():
            raise FileNotFoundError(f"Checkpoint root directory does not exist: {root}")

        folders = sorted(
            (
                folder for folder in root.iterdir()
                if folder.is_dir()
                and any(
                    path.is_file() and path.suffix.lower() in self.IMAGE_SUFFIXES
                    for path in folder.iterdir()
                )
            ),
            key=lambda folder: folder.name.casefold(),
        )
        if not folders:
            raise FileNotFoundError(f"No checkpoint folders with images found in: {root}")
        return folders

    @staticmethod
    def _load_config_bindings(config_path):
        if not config_path.is_file():
            raise FileNotFoundError(f"Game config file not found: {config_path}")

        pattern = re.compile(r'^\s*bind\s+"([^"]+)"\s+"([^"]+)"', re.IGNORECASE)
        bindings = []
        for line in config_path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = pattern.match(line)
            if match:
                key, command = match.group(1), match.group(2)
                if key.upper() in HalfLifeEnv.BLOCKED_BIND_KEYS:
                    continue
                command_name = command.lstrip("+").strip().casefold()
                if any(
                    command_name == blocked or command_name.startswith(blocked + " ")
                    for blocked in HalfLifeEnv.BLOCKED_BIND_COMMANDS
                ):
                    continue
                bindings.append((key, command))
        if not bindings:
            raise ValueError(f"No bind entries found in game config: {config_path}")
        return bindings

    def _load_checkpoint_images(self, folders):
        checkpoints = []
        for folder in folders:
            image_paths = sorted(
                path for path in folder.iterdir()
                if path.is_file() and path.suffix.lower() in self.IMAGE_SUFFIXES
            )
            if not image_paths:
                raise FileNotFoundError(f"No checkpoint images found in: {folder}")

            references = []
            for image_path in image_paths:
                image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    raise ValueError(f"Could not read checkpoint image: {image_path}")
                target_size = (self.monitor["width"], self.monitor["height"])
                if (image.shape[1], image.shape[0]) != target_size:
                    image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
                references.append(image)
            checkpoints.append(references)
        return checkpoints

    def _capture_frame(self):
        screenshot = np.asarray(self.sct.grab(self.monitor))
        rgb = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2RGB)
        grayscale = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        depth = self.depth_estimator(rgb)
        return grayscale, depth

    def _capture_loop(self):
        """Capture frames continuously, independently of the agent's actions."""
        next_capture = time.monotonic()
        while not self._capture_stop.is_set():
            try:
                grayscale, depth = self._capture_frame()
                captured_at = time.monotonic()
                with self._frame_history_lock:
                    self._frame_history.append((captured_at, (grayscale, depth)))
            except Exception as error:
                print(f"Screenshot capture failed: {error}")

            next_capture += 0.1
            delay = max(0.0, next_capture - time.monotonic())
            self._capture_stop.wait(delay)

    def _get_observation(self, reset_history=False):
        while True:
            with self._frame_history_lock:
                if self._frame_history:
                    break
            if self._capture_stop.wait(0.01):
                raise RuntimeError("Screenshot capture stopped before the first frame.")

        with self._frame_history_lock:
            timestamp, frame = self._frame_history[-1]
            if reset_history:
                self._frame_history.clear()
                self._frame_history.extend(
                    (timestamp - offset, frame) for offset in (3.0, 2.0, 1.0, 0.0)
                )

            # Channels are ordered oldest to newest: approximately -3s, -2s, -1s, now.
            selected_frames = []
            for offset in (3.0, 2.0, 1.0, 0.0):
                target_time = timestamp - offset
                frame = next(
                    (
                        candidate
                        for frame_time, candidate in reversed(self._frame_history)
                        if frame_time <= target_time
                    ),
                    self._frame_history[0][1],
                )
                selected_frames.extend(frame)

            cutoff = timestamp - 3.0
            while len(self._frame_history) > 2 and self._frame_history[1][0] < cutoff:
                self._frame_history.popleft()

        return np.stack(selected_frames, axis=-1)

    def _similarity(self, grayscale, references):
        # Each folder may contain different camera angles; use its best matching view.
        return max(
            float(cv2.matchTemplate(grayscale, reference, cv2.TM_CCOEFF_NORMED)[0, 0])
            for reference in references
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._release_keys()
        self._reload_map()
        self.checkpoint_index = 0
        self._claimed_checkpoints.clear()
        self._previous_similarity = 0.0
        self._checkpoint_hits = [0] * len(self.checkpoint_images)
        self._recent_actions.clear()
        self._stuck_hits = 0
        self._is_stuck = False
        self._episode_steps = 0
        self._episode_id = str(time.time_ns())
        self._episode_actions = []
        self._start_behavior_capture()
        self.footstep_detector.discard_pending_events()
        observation = self._get_observation(reset_history=True)
        self._last_observation = observation.copy()
        return observation, {"checkpoint_index": self.checkpoint_index}

    @staticmethod
    def _reload_map():
        """Restart c1a0 so every episode begins from the same starting point."""
        pydirectinput.press("`")
        time.sleep(0.5)
        pydirectinput.write("map c1a0", interval=0.05)
        pydirectinput.press("enter")
        time.sleep(3)

    def _start_behavior_capture(self):
        if self._behavior_temp is not None:
            self._behavior_temp.cleanup()
        pending_root = self._behavior_root / ".pending"
        pending_root.mkdir(parents=True, exist_ok=True)
        self._behavior_temp = tempfile.TemporaryDirectory(
            prefix=f"episode_{self._episode_id}_", dir=str(pending_root)
        )
        self._episode_capture_dir = Path(self._behavior_temp.name)

    def _save_observation_frame(self, observation):
        # Preserve all eight grayscale/depth channels as a left-to-right strip.
        frame_strip = np.concatenate(
            [observation[:, :, channel] for channel in range(observation.shape[2])],
            axis=1,
        )
        success, encoded = cv2.imencode(
            ".jpg", frame_strip, [cv2.IMWRITE_JPEG_QUALITY, 85]
        )
        if not success:
            print(f"Could not encode observation for step {self._episode_steps}.")
            return None

        filename = f"step_{self._episode_steps:06d}.jpg"
        observation_dir = self._episode_capture_dir / "observations"
        try:
            observation_dir.mkdir(parents=True, exist_ok=True)
            (observation_dir / filename).write_bytes(encoded.tobytes())
        except OSError as error:
            print(f"Could not save observation for step {self._episode_steps}: {error}")
            return None
        return f"episodes/{self._episode_id}/observations/{filename}"

    def _persist_episode_capture(self):
        final_dir = self._behavior_root / "episodes" / self._episode_id
        if self._behavior_temp is not None:
            final_dir.parent.mkdir(parents=True, exist_ok=True)
            self._episode_capture_dir.replace(final_dir)
            self._behavior_temp.cleanup()
            self._behavior_temp = None
        else:
            final_dir.mkdir(parents=True, exist_ok=True)
        self._episode_capture_dir = final_dir

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(
                f"Invalid action {action}; expected an integer from 0 to {self.action_space.n - 1}"
            )

        action = int(action)
        action_observation = self._last_observation
        self._episode_steps += 1
        if self._is_turn_action(action):
            duration = self.turn_action_duration
        elif action == 5:
            duration = self.crouch_jump_duration
        elif action in self.JUMP_ACTIONS:
            duration = self.jump_action_duration
        else:
            duration = self.action_duration

        action_started = time.monotonic()
        self._take_action(action)
        self._recent_actions.append((action_started, duration, action))
        time.sleep(duration)
        observation = self._get_observation()
        # Do not let an action remain held while PPO processes the next observation.
        self._release_keys()
        grayscale = observation[:, :, -2]
        now = time.monotonic()
        footstep_detected, footstep_similarity = self.footstep_detector.consume_new_event()

        movement_seconds = self._recent_movement_seconds(now)
        frame_similarities = [
            float(
                cv2.matchTemplate(
                    observation[:, :, frame_index * 2],
                    observation[:, :, (frame_index + 1) * 2],
                    cv2.TM_CCOEFF_NORMED,
                )[0, 0]
            )
            for frame_index in range(3)
        ]
        visual_similarity = float(np.mean(frame_similarities))
        stuck_detected = self._update_stuck_state(movement_seconds, visual_similarity)

        reward = 0.0
        if footstep_detected and self._is_movement_action(action):
            reward += self.footstep_reward
        if stuck_detected:
            reward -= self.stuck_penalty
        checkpoint_advanced = False
        checkpoint_scores = {}
        for checkpoint_number, references in enumerate(self.checkpoint_images):
            if checkpoint_number in self._claimed_checkpoints:
                continue
            score = self._similarity(grayscale, references)
            checkpoint_scores[checkpoint_number] = score
            if score >= self.checkpoint_threshold:
                self._checkpoint_hits[checkpoint_number] += 1
            else:
                self._checkpoint_hits[checkpoint_number] = 0

        best_checkpoint_number = None
        best_checkpoint_score = None
        if checkpoint_scores:
            best_checkpoint_number = max(checkpoint_scores, key=checkpoint_scores.get)
            best_checkpoint_score = checkpoint_scores[best_checkpoint_number]
            reward += 0.25 * (best_checkpoint_score - self._previous_similarity)
            self._previous_similarity = best_checkpoint_score

        confirmed_checkpoints = [
            checkpoint_number
            for checkpoint_number, hit_count in enumerate(self._checkpoint_hits)
            if checkpoint_number not in self._claimed_checkpoints
            and hit_count >= self.checkpoint_confirm_frames
        ]
        triggered_checkpoint = max(confirmed_checkpoints) if confirmed_checkpoints else None
        checkpoints_claimed = []
        if triggered_checkpoint is not None:
            checkpoints_claimed = [
                checkpoint_number
                for checkpoint_number in range(triggered_checkpoint + 1)
                if checkpoint_number not in self._claimed_checkpoints
            ]
            self._claimed_checkpoints.update(checkpoints_claimed)
            for checkpoint_number in checkpoints_claimed:
                self._checkpoint_hits[checkpoint_number] = 0
            reward += float(len(checkpoints_claimed))
            checkpoint_advanced = True
            self.checkpoint_index = len(self._claimed_checkpoints)
            self._previous_similarity = 0.0

        # Checkpoints are sorted by folder name; reaching d credits earlier nodes.
        terminated = len(self._claimed_checkpoints) >= self.termination_checkpoint_count
        truncated = not terminated and self._episode_steps >= self.MAX_EPISODE_STEPS
        episode_score = None
        time_coefficient = None
        if terminated or truncated:
            checkpoint_points = len(self._claimed_checkpoints)
            if terminated:
                time_coefficient = 1.0 + self.SUCCESS_TIME_BONUS * (
                    1.0 - self._episode_steps / self.MAX_EPISODE_STEPS
                )
                episode_score = checkpoint_points * time_coefficient
            else:
                episode_score = (
                    checkpoint_points * self.INCOMPLETE_SCORE_FACTOR
                    - self.INCOMPLETE_EPISODE_PENALTY
                )
                reward -= self.INCOMPLETE_EPISODE_PENALTY
            if terminated:
                # Checkpoint points were already rewarded when claimed. Add only
                # the speed bonus so progress and exploration rewards remain intact.
                reward += checkpoint_points * (time_coefficient - 1.0)

        info = {
            "checkpoint_index": self.checkpoint_index,
            "checkpoint_score": best_checkpoint_score,
            "best_checkpoint_index": (
                best_checkpoint_number + 1 if best_checkpoint_number is not None else None
            ),
            "checkpoint_advanced": checkpoint_advanced,
            "checkpoint_triggered": (
                triggered_checkpoint + 1 if triggered_checkpoint is not None else None
            ),
            "checkpoints_claimed": [number + 1 for number in checkpoints_claimed],
            "is_success": terminated,
            "is_stuck": self._is_stuck,
            "stuck_detected": stuck_detected,
            "visual_similarity_3s": visual_similarity,
            "movement_attempt_seconds": movement_seconds,
            "action_name": self._action_name(action),
            "footstep_detected": footstep_detected,
            "footstep_similarity": footstep_similarity,
            "episode_steps": self._episode_steps,
            "checkpoint_points": len(self._claimed_checkpoints),
            "time_coefficient": time_coefficient,
            "episode_final_score": episode_score,
        }
        observation_file = (
            self._save_observation_frame(action_observation)
            if action_observation is not None
            else None
        )
        self._episode_actions.append(
            {
                "step": self._episode_steps,
                "action_id": action,
                "action": info["action_name"],
                "duration_seconds": duration,
                "reward": float(reward),
                "best_checkpoint_index": info["best_checkpoint_index"],
                "checkpoint_similarity": best_checkpoint_score,
                "movement_attempt_seconds_last_3s": movement_seconds,
                "stuck_detected": stuck_detected,
                "footstep_detected": footstep_detected,
                "observation_file": observation_file,
            }
        )
        for checkpoint_number in checkpoints_claimed:
            self._record_checkpoint_behavior(
                checkpoint_number=checkpoint_number,
                triggered_checkpoint=triggered_checkpoint,
                final_score=episode_score,
            )
        self._last_observation = observation.copy()
        return observation, float(reward), terminated, truncated, info

    def _record_checkpoint_behavior(
        self, checkpoint_number, triggered_checkpoint, final_score
    ):
        checkpoint_folder = self.checkpoint_folders[checkpoint_number]
        checkpoint_name = checkpoint_folder.name
        output_dir = (
            Path(__file__).resolve().parent
            / "successful_behaviors"
            / f"checkpoint_{checkpoint_number + 1:02d}_{checkpoint_name}"
        )
        output_path = output_dir / f"episode_{self._episode_id}.json"
        record = {
            "map": "c1a0",
            "episode_id": self._episode_id,
            "checkpoint_index": checkpoint_number + 1,
            "checkpoint_name": checkpoint_name,
            "triggered_checkpoint_index": (
                triggered_checkpoint + 1 if triggered_checkpoint is not None else None
            ),
            "steps_to_checkpoint": self._episode_steps,
            "checkpoint_points_at_trigger": len(self._claimed_checkpoints),
            "episode_final_score_at_trigger": final_score,
            "observation_format": (
                "JPEG strip: four grayscale/depth frame pairs arranged oldest-to-newest; "
                "split into eight equal-width panels, alternating grayscale then relative depth"
            ),
            "actions_from_episode_start": list(self._episode_actions),
        }

        print(
            f"通过检查点: {checkpoint_name}（检查点 {checkpoint_number + 1}），"
            f"第 {self._episode_steps} 步"
        )
        try:
            self._persist_episode_capture()
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"有效行为已记录到: {output_path}")
        except OSError as error:
            print(f"记录检查点 {checkpoint_name} 的有效行为失败: {error}")

    def _action_name(self, action):
        if action < self.BASE_ACTION_COUNT:
            return f"macro:{action}"
        key, command = self.config_bindings[action - self.BASE_ACTION_COUNT]
        return f"bind:{key} -> {command}"

    def _is_turn_action(self, action):
        if action in {2, 3}:
            return True
        if action < self.BASE_ACTION_COUNT:
            return False
        key, command = self.config_bindings[action - self.BASE_ACTION_COUNT]
        return (
            key.upper() in self.DIRECTION_BIND_KEYS
            or command.casefold() in {"+left", "+right"}
        )

    def _recent_movement_seconds(self, now):
        window_start = now - 3.0
        while self._recent_actions and self._recent_actions[0][0] + self._recent_actions[0][1] < window_start:
            self._recent_actions.popleft()

        attempted_seconds = 0.0
        for started, duration, action in self._recent_actions:
            if not self._is_movement_action(action):
                continue
            overlap_start = max(started, window_start)
            overlap_end = min(started + duration, now)
            attempted_seconds += max(0.0, overlap_end - overlap_start)
        return attempted_seconds

    def _is_movement_action(self, action):
        if action in self.MOVEMENT_ACTIONS:
            return True
        if action < self.BASE_ACTION_COUNT:
            return False
        _, command = self.config_bindings[action - self.BASE_ACTION_COUNT]
        return command in {"+forward", "+back", "+moveleft", "+moveright"}

    def _update_stuck_state(self, movement_seconds, visual_similarity):
        movement_attempted = movement_seconds >= self.stuck_min_movement_seconds
        visually_stagnant = visual_similarity >= self.stuck_similarity_threshold

        if movement_attempted and visually_stagnant:
            self._stuck_hits += 1
        else:
            self._stuck_hits = 0

        if self._is_stuck and not visually_stagnant:
            self._is_stuck = False

        if self._stuck_hits >= self.stuck_confirm_steps and not self._is_stuck:
            self._is_stuck = True
            return True
        return False

    def _release_keys(self):
        for key in tuple(self._held_keys):
            pydirectinput.keyUp(key)
        for button in tuple(self._held_mouse_buttons):
            pydirectinput.mouseUp(button=button)
        self._held_keys.clear()
        self._held_mouse_buttons.clear()

    def _key_down(self, key):
        pydirectinput.keyDown(key)
        self._held_keys.add(key)

    def _press_key_sequence(self, keys):
        for key in keys:
            pydirectinput.keyDown(key)
        for key in reversed(keys):
            pydirectinput.keyUp(key)

    def _execute_config_binding(self, binding_index):
        key_name, command = self.config_bindings[binding_index]

        if key_name in {"MWHEELDOWN", "MWHEELUP"}:
            # pydirectinput does not expose a scroll() function. Send the
            # standard Windows mouse-wheel event directly (one wheel notch).
            wheel_delta = -120 if key_name == "MWHEELDOWN" else 120
            ctypes.windll.user32.mouse_event(0x0800, 0, 0, wheel_delta, 0)
            return

        if key_name in {"MOUSE1", "MOUSE2"}:
            button = "left" if key_name == "MOUSE1" else "right"
            if command.startswith("+"):
                pydirectinput.mouseDown(button=button)
                self._held_mouse_buttons.add(button)
            else:
                pydirectinput.click(button=button)
            return

        key = self.KEY_ALIASES.get(key_name, key_name.lower())
        keys = key if isinstance(key, tuple) else (key,)
        if command.startswith("+"):
            for item in keys:
                self._key_down(item)
        else:
            self._press_key_sequence(keys)

    def _take_action(self, action):
        self._release_keys()
        if action >= self.BASE_ACTION_COUNT:
            self._execute_config_binding(action - self.BASE_ACTION_COUNT)
        elif action == 2:
            self._key_down("left")
        elif action == 3:
            self._key_down("right")
        else:
            for key in self.ACTION_KEYS[action]:
                self._key_down(key)

    def close(self):
        if getattr(self, "_behavior_temp", None) is not None:
            self._behavior_temp.cleanup()
            self._behavior_temp = None
        if getattr(self, "_capture_stop", None) is not None:
            self._capture_stop.set()
        if getattr(self, "_capture_thread", None) is not None:
            self._capture_thread.join(timeout=2.0)
            self._capture_thread = None
        if getattr(self, "footstep_detector", None) is not None:
            self.footstep_detector.close()
            self.footstep_detector = None
        if getattr(self, "sct", None) is not None:
            self._release_keys()
            self.sct.close()
            self.sct = None
