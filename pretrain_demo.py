"""Behavior-clone a saved PPO policy from record_demo.py demonstrations.

This updates the policy from labeled screen/action pairs. It does not step the
game environment. The resulting PPO archive can later be continued with main.py.
"""

import argparse
import bisect
import json
import math
from collections import Counter
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from hl1_env import HalfLifeEnv
from main import EpisodeCompletePPO
from depth_perception import RelativeDepthEstimator


PROJECT_ROOT = Path(__file__).resolve().parent


class PolicyShapeEnv(gym.Env):
    """A no-input environment used only to attach a saved model's spaces."""

    metadata = {"render_modes": []}

    def __init__(self, action_count):
        super().__init__()
        self.action_space = spaces.Discrete(action_count)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(480, 640, 8), dtype=np.uint8
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(self.observation_space.shape, dtype=np.uint8), {}

    def step(self, action):
        return (
            np.zeros(self.observation_space.shape, dtype=np.uint8),
            0.0,
            False,
            False,
            {},
        )


class DemoDataset(Dataset):
    def __init__(self, images, sequences, labels):
        self.images = images
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # Each timestep contributes grayscale and relative depth channels.
        observation = np.concatenate(self.images[self.sequences[index]], axis=0)
        return torch.from_numpy(observation), int(self.labels[index])


def normalize_controls(keys):
    aliases = {
        "LCTRL": "CTRL", "RCTRL": "CTRL", "LSHIFT": "SHIFT", "RSHIFT": "SHIFT",
        "LALT": "ALT", "RALT": "ALT", "MOUSE_LEFT": "MOUSE1",
        "MOUSE_RIGHT": "MOUSE2", "MOUSE_MIDDLE": "MOUSE3", "UP": "UPARROW",
        "DOWN": "DOWNARROW", "LEFT": "LEFTARROW", "RIGHT": "RIGHTARROW",
        "PAGEUP": "PGUP", "PAGEDOWN": "PGDN",
    }
    return frozenset(aliases.get(str(key).upper(), str(key).upper()) for key in keys)


def build_action_mapper():
    macro_actions = {
        frozenset(key.upper() for key in keys): int(action)
        for action, keys in HalfLifeEnv.ACTION_KEYS.items()
    }
    macro_actions[frozenset({"LEFTARROW"})] = 2
    macro_actions[frozenset({"RIGHTARROW"})] = 3

    config_path = PROJECT_ROOT / "config.cfg"
    bindings = HalfLifeEnv._load_config_bindings(config_path)
    single_actions = {}
    for index, (key, _command) in enumerate(bindings):
        normalized = normalize_controls([key])
        if len(normalized) == 1:
            single_actions.setdefault(next(iter(normalized)), HalfLifeEnv.BASE_ACTION_COUNT + index)

    def map_keys(keys):
        normalized = normalize_controls(keys)
        if normalized in macro_actions:
            return macro_actions[normalized]
        if len(normalized) == 1:
            return single_actions.get(next(iter(normalized)))
        return None

    mouse_actions = {
        "left": 2,
        "right": 3,
        "up": single_actions.get("PGUP"),
        "down": single_actions.get("PGDN"),
    }
    return map_keys, mouse_actions, len(bindings) + HalfLifeEnv.BASE_ACTION_COUNT


def load_demonstrations(demo_dir, map_keys, mouse_actions, mouse_threshold):
    metadata_path = demo_dir / "metadata.json"
    input_path = demo_dir / "inputs.jsonl"
    if not metadata_path.is_file() or not input_path.is_file():
        raise FileNotFoundError(f"Not a recorder output directory: {demo_dir}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    region = metadata.get("capture_region", {})
    if (region.get("width"), region.get("height")) != (640, 480):
        raise ValueError("The demo must use the training capture size of 640x480.")

    rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line]
    frames = [row for row in rows if row.get("type") == "frame"]
    if len(frames) < 4:
        raise ValueError("The demo has too few recorded frames.")

    times = [float(row["t"]) for row in frames]
    image_cache = []
    depth_estimator = RelativeDepthEstimator()
    for row in frames:
        image_path = demo_dir / row["frame"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (480, 640):
            raise ValueError(f"Missing or invalid demo frame: {image_path}")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        depth = depth_estimator(rgb)
        image_cache.append(np.stack((gray, depth), axis=0))
    image_cache = np.stack(image_cache, axis=0)

    sequences, labels = [], []
    skipped = Counter()
    previous_mouse = None
    for index, row in enumerate(frames):
        keys = row.get("keys_down", [])
        action = map_keys(keys)
        current_mouse = (row.get("mouse_x"), row.get("mouse_y"))
        if (
            action == 0
            and not keys
            and previous_mouse is not None
            and current_mouse[0] is not None
            and current_mouse[1] is not None
        ):
            dx = int(current_mouse[0]) - int(previous_mouse[0])
            dy = int(current_mouse[1]) - int(previous_mouse[1])
            if max(abs(dx), abs(dy)) >= mouse_threshold:
                if abs(dx) >= abs(dy):
                    action = mouse_actions["right"] if dx > 0 else mouse_actions["left"]
                else:
                    action = mouse_actions["down"] if dy > 0 else mouse_actions["up"]
        if current_mouse[0] is not None and current_mouse[1] is not None:
            previous_mouse = current_mouse
        if action is None:
            label = "+".join(sorted(normalize_controls(keys))) or "<empty>"
            skipped[label] += 1
            continue

        now = times[index]
        history = []
        for offset in (3.0, 2.0, 1.0, 0.0):
            frame_index = bisect.bisect_right(times, now - offset) - 1
            history.append(max(0, frame_index))
        sequences.append(history)
        labels.append(action)

    if not labels:
        raise ValueError("No demo actions could be mapped to the current PPO action space.")
    return DemoDataset(image_cache, np.asarray(sequences, dtype=np.int64),
                       np.asarray(labels, dtype=np.int64)), skipped


def parse_args():
    parser = argparse.ArgumentParser(description="Supervised PPO policy pretraining from recorded gameplay.")
    parser.add_argument("--demo", type=Path, required=True, help="A demo_... directory from record_demo.py.")
    parser.add_argument("--source", type=Path,
                        default=PROJECT_ROOT / "models" / "c1a0_ppo_rgbd_midas.zip",
                        help="Existing PPO checkpoint used to initialize the policy.")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "models" / "c1a0_ppo_rgbd_midas_demo_bc.zip",
                        help="Save the behavior-cloned PPO checkpoint here.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--mouse-threshold-px", type=int, default=4,
                        help="Minimum recorded cursor movement to infer a turn/look action.")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0 or args.mouse_threshold_px < 1:
        parser.error("epochs, batch-size, mouse-threshold-px, and learning-rate must be positive")
    return args


def main():
    args = parse_args()
    demo_dir = args.demo if args.demo.is_absolute() else PROJECT_ROOT / args.demo
    source_path = args.source if args.source.is_absolute() else PROJECT_ROOT / args.source
    output_path = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    if not source_path.is_file():
        raise FileNotFoundError(f"PPO source checkpoint not found: {source_path}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Use --device cpu or repair the PyTorch CUDA setup.")

    map_keys, mouse_actions, action_count = build_action_mapper()
    dataset, skipped = load_demonstrations(
        demo_dir, map_keys, mouse_actions, args.mouse_threshold_px
    )
    label_counts = Counter(int(label) for label in dataset.labels)
    weights = [1.0 / math.sqrt(label_counts[int(label)]) for label in dataset.labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0)

    model_env = PolicyShapeEnv(action_count)
    model = EpisodeCompletePPO.load(str(source_path), env=model_env, device=args.device)
    try:
        expected_shape = tuple(model.observation_space.shape)
        if expected_shape not in ((8, 480, 640), (480, 640, 8)):
            raise ValueError(f"The source model has an unexpected observation shape: {expected_shape}")
        if model.action_space.n != action_count:
            raise ValueError(
                f"Demo action count ({action_count}) does not match source model "
                f"({model.action_space.n}); config.cfg or the checkpoint may have changed."
            )

        model.policy.set_training_mode(True)
        optimizer = torch.optim.Adam(
            model.policy.parameters(), lr=args.learning_rate, weight_decay=1e-5
        )
        model.policy.optimizer = optimizer
        print(f"Demo samples mapped: {len(dataset)} / {len(dataset) + sum(skipped.values())}")
        print(f"Mapped action classes: {dict(sorted(label_counts.items()))}")
        if skipped:
            print(f"Unmapped key combinations skipped: {dict(skipped)}")

        for epoch in range(1, args.epochs + 1):
            total_loss = 0.0
            total_correct = 0
            total_seen = 0
            for observations, labels in loader:
                observations = observations.to(args.device)
                labels = labels.to(args.device, dtype=torch.long)
                distribution = model.policy.get_distribution(observations)
                logits = distribution.distribution.logits
                loss = torch.nn.functional.cross_entropy(logits, labels)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.policy.parameters(), max_norm=1.0)
                optimizer.step()

                batch_size = labels.shape[0]
                total_loss += float(loss.detach()) * batch_size
                total_correct += int((logits.argmax(dim=1) == labels).sum())
                total_seen += batch_size

            print(
                f"Epoch {epoch}/{args.epochs}: "
                f"loss={total_loss / total_seen:.4f}, "
                f"accuracy={total_correct / total_seen:.3f}"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(output_path))
        info_path = output_path.with_name(f"{output_path.stem}_demo_info.json")
        info_path.write_text(json.dumps({
            "source_model": str(source_path),
            "demo_directory": str(demo_dir),
            "samples_used": len(dataset),
            "skipped_key_combinations": dict(skipped),
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "mouse_threshold_px": args.mouse_threshold_px,
            "device": args.device,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Behavior-cloned model saved to: {output_path}")
    finally:
        model_env.close()


if __name__ == "__main__":
    main()
