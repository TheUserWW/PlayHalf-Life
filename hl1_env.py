from pathlib import Path
from collections import deque
import time

import cv2
import gymnasium as gym
import mss
import numpy as np
import pydirectinput
from gymnasium import spaces


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

    def __init__(
        self,
        monitor=None,
        action_duration=0.1,
        jump_action_duration=0.35,
        crouch_jump_duration=0.35,
        checkpoint_threshold=0.75,
        checkpoint_confirm_frames=3,
        stuck_similarity_threshold=0.90,
        stuck_min_movement_seconds=1.5,
        stuck_confirm_steps=3,
        stuck_penalty=1.0,
        checkpoint_root=None,
    ):
        super().__init__()
        # Keep multi-key actions near-simultaneous; action timing is controlled below.
        pydirectinput.PAUSE = 0
        # 2 and 3 turn; remaining IDs are defined by ACTION_KEYS.
        self.action_space = spaces.Discrete(17)
        self.monitor = monitor or {"top": 0, "left": 0, "width": 640, "height": 480}
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.monitor["height"], self.monitor["width"], 4),
            dtype=np.uint8,
        )
        # Keep enough frames to select one view from each second of the last 3 seconds.
        self._frame_history = deque(maxlen=128)
        self.action_duration = action_duration
        self.jump_action_duration = jump_action_duration
        self.crouch_jump_duration = crouch_jump_duration
        self.checkpoint_threshold = checkpoint_threshold
        self.checkpoint_confirm_frames = checkpoint_confirm_frames
        self.stuck_similarity_threshold = stuck_similarity_threshold
        self.stuck_min_movement_seconds = stuck_min_movement_seconds
        self.stuck_confirm_steps = stuck_confirm_steps
        self.stuck_penalty = stuck_penalty
        self._recent_actions = deque(maxlen=128)
        self._stuck_hits = 0
        self._is_stuck = False
        self.sct = mss.mss()

        root = Path(checkpoint_root) if checkpoint_root else Path(__file__).resolve().parent / "rewards"
        self.checkpoint_folders = self._discover_checkpoint_folders(root)
        self.checkpoint_images = self._load_checkpoint_images(self.checkpoint_folders)
        self.checkpoint_index = 0
        self._previous_similarity = 0.0
        self._checkpoint_hits = 0

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

    def _capture_grayscale(self):
        screenshot = np.asarray(self.sct.grab(self.monitor))
        return cv2.cvtColor(screenshot, cv2.COLOR_BGRA2GRAY)

    def _get_observation(self, reset_history=False):
        grayscale = self._capture_grayscale()
        timestamp = time.monotonic()

        if reset_history:
            self._frame_history.clear()
            self._frame_history.extend(
                (timestamp - offset, grayscale) for offset in (3.0, 2.0, 1.0, 0.0)
            )
        else:
            self._frame_history.append((timestamp, grayscale))

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
            selected_frames.append(frame)

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
        self.checkpoint_index = 0
        self._previous_similarity = 0.0
        self._checkpoint_hits = 0
        self._recent_actions.clear()
        self._stuck_hits = 0
        self._is_stuck = False
        return self._get_observation(reset_history=True), {"checkpoint_index": self.checkpoint_index}

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}; expected an integer from 0 to 16")

        action = int(action)
        if action == 5:
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
        grayscale = observation[:, :, -1]
        now = time.monotonic()

        movement_seconds = self._recent_movement_seconds(now)
        frame_similarities = [
            float(
                cv2.matchTemplate(
                    observation[:, :, frame_index],
                    observation[:, :, frame_index + 1],
                    cv2.TM_CCOEFF_NORMED,
                )[0, 0]
            )
            for frame_index in range(3)
        ]
        visual_similarity = float(np.mean(frame_similarities))
        stuck_detected = self._update_stuck_state(movement_seconds, visual_similarity)

        reward = 0.0
        if stuck_detected:
            reward -= self.stuck_penalty
        checkpoint_advanced = False
        checkpoint_score = None
        if self.checkpoint_index < len(self.checkpoint_images):
            references = self.checkpoint_images[self.checkpoint_index]
            checkpoint_score = self._similarity(grayscale, references)
            reward += 0.25 * (checkpoint_score - self._previous_similarity)
            self._previous_similarity = checkpoint_score

            if checkpoint_score >= self.checkpoint_threshold:
                self._checkpoint_hits += 1
            else:
                self._checkpoint_hits = 0

            if self._checkpoint_hits >= self.checkpoint_confirm_frames:
                reward += 1.0
                self.checkpoint_index += 1
                checkpoint_advanced = True
                self._previous_similarity = 0.0
                self._checkpoint_hits = 0

        # Reaching the final discovered checkpoint completes this c1a0 run.
        terminated = checkpoint_advanced and self.checkpoint_index == len(self.checkpoint_images)
        truncated = False
        info = {
            "checkpoint_index": self.checkpoint_index,
            "checkpoint_score": checkpoint_score,
            "checkpoint_advanced": checkpoint_advanced,
            "is_success": terminated,
            "is_stuck": self._is_stuck,
            "stuck_detected": stuck_detected,
            "visual_similarity_3s": visual_similarity,
            "movement_attempt_seconds": movement_seconds,
        }
        return observation, float(reward), terminated, truncated, info

    def _recent_movement_seconds(self, now):
        window_start = now - 3.0
        while self._recent_actions and self._recent_actions[0][0] + self._recent_actions[0][1] < window_start:
            self._recent_actions.popleft()

        attempted_seconds = 0.0
        for started, duration, action in self._recent_actions:
            if action not in self.MOVEMENT_ACTIONS:
                continue
            overlap_start = max(started, window_start)
            overlap_end = min(started + duration, now)
            attempted_seconds += max(0.0, overlap_end - overlap_start)
        return attempted_seconds

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
        for key in ("w", "a", "s", "d", "ctrl", "space", "left", "right"):
            pydirectinput.keyUp(key)

    def _take_action(self, action):
        self._release_keys()
        if action == 2:
            pydirectinput.press("left")
        elif action == 3:
            pydirectinput.press("right")
        else:
            for key in self.ACTION_KEYS[action]:
                pydirectinput.keyDown(key)

    def close(self):
        if getattr(self, "sct", None) is not None:
            self._release_keys()
            self.sct.close()
            self.sct = None
