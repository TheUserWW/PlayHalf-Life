import argparse
import atexit
from pathlib import Path
import sys
import time
from datetime import datetime

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.utils import get_schedule_fn, obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3 import PPO
import torch
import pydirectinput

from hl1_env import HalfLifeEnv


MODEL_PATH = Path(__file__).resolve().parent / "models" / "c1a0_ppo_rgbd_midas"
DEVICE = "cuda"


class Tee:
    """Write console output to both the terminal and a UTF-8 log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, message):
        for stream in self.streams:
            stream.write(message)
            stream.flush()
        return len(message)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)

    @property
    def encoding(self):
        return getattr(self.streams[0], "encoding", "utf-8")


def start_training_log():
    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_path = log_dir / f"training_{timestamp}.log"
    log_file = log_path.open("w", encoding="utf-8", buffering=1)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = Tee(original_stdout, log_file)
    sys.stderr = Tee(original_stderr, log_file)
    print(f"Training log: {log_path}")

    def close_log():
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()

    atexit.register(close_log)


def reset_game_to_c1a0():
    """Open the in-game console and reload the c1a0 map after training."""
    print("Reloading c1a0 through the game console...")
    pydirectinput.PAUSE = 0.05
    pydirectinput.press("`")
    time.sleep(0.5)
    pydirectinput.write("map c1a0", interval=0.05)
    pydirectinput.press("enter")
    time.sleep(3)


class TrainingScoreCallback(BaseCallback):
    """Collect episode returns and total reward for this training run."""

    def __init__(self, checkpoint_threshold, checkpoint_names):
        super().__init__(verbose=0)
        self.checkpoint_threshold = checkpoint_threshold
        self.checkpoint_names = checkpoint_names
        self.training_score = 0.0
        self.episode_scores = []
        self.episode_final_scores = []
        self.current_episode_scores = []
        self.passed_checkpoints = []
        self.triggered_checkpoints = []
        self.best_checkpoint_matches = []
        self.footstep_events = 0

    def _on_training_start(self):
        self.current_episode_scores = [0.0] * self.training_env.num_envs
        self.passed_checkpoints = [set() for _ in range(self.training_env.num_envs)]
        self.triggered_checkpoints = [set() for _ in range(self.training_env.num_envs)]
        self.best_checkpoint_matches = [None] * self.training_env.num_envs

    def _on_step(self):
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]
        infos = self.locals["infos"]
        for index, reward in enumerate(rewards):
            reward = float(reward)
            self.training_score += reward
            self.current_episode_scores[index] += reward
            if infos[index].get("footstep_detected"):
                self.footstep_events += 1
            checkpoint_score = infos[index].get("checkpoint_score")
            if checkpoint_score is not None:
                checkpoint_number = infos[index].get("best_checkpoint_index")
                if checkpoint_number is not None:
                    match = (float(checkpoint_score), int(checkpoint_number))
                    previous_best = self.best_checkpoint_matches[index]
                    if previous_best is None or match[0] > previous_best[0]:
                        self.best_checkpoint_matches[index] = match
            for checkpoint_number in infos[index].get("checkpoints_claimed", []):
                self.passed_checkpoints[index].add(int(checkpoint_number))
            triggered_checkpoint = infos[index].get("checkpoint_triggered")
            if triggered_checkpoint is not None:
                self.triggered_checkpoints[index].add(int(triggered_checkpoint))
            if dones[index]:
                self.episode_scores.append(self.current_episode_scores[index])
                final_score = infos[index].get("episode_final_score")
                if final_score is not None:
                    self.episode_final_scores.append(float(final_score))
                end_reason = "通过检查点 d" if infos[index].get("is_success") else "达到 500 步上限"
                print(f"本次训练尝试结束：{end_reason}（第 {infos[index].get('episode_steps')} 步）")
                self.current_episode_scores[index] = 0.0
        # End each invocation at checkpoint d or the environment's step limit.
        return not any(dones)

    def print_summary(self):
        if self.episode_final_scores:
            print(f"Last episode final score: {self.episode_final_scores[-1]:.3f}")
        print(f"本轮训练累计得分（所有训练步奖励之和）: {self.training_score:.3f}")
        print(f"本轮检测到的脚步声事件: {self.footstep_events}")
        if self.episode_scores:
            print(f"已完成回合数: {len(self.episode_scores)}")
            print(f"最后一个完整回合训练奖励和: {self.episode_scores[-1]:.3f}")
        else:
            print("本轮没有完成完整回合。")
        for index, checkpoints in enumerate(self.passed_checkpoints, start=1):
            if checkpoints:
                checkpoint_list = ", ".join(
                    self._checkpoint_label(number) for number in sorted(checkpoints)
                )
                print(f"环境 {index} 本轮通过的检查点: {checkpoint_list}")
            else:
                print(f"环境 {index} 本轮通过的检查点: 无")
            triggered = self.triggered_checkpoints[index - 1]
            if triggered:
                trigger_list = ", ".join(
                    self._checkpoint_label(number) for number in sorted(triggered)
                )
                print(f"环境 {index} 本轮实际触发节点: {trigger_list}")
            best_match = self.best_checkpoint_matches[index - 1]
            if best_match is not None:
                print(
                    f"环境 {index} 本轮最高匹配度: {best_match[0]:.3f} "
                    f"({self._checkpoint_label(best_match[1])}, 通过阈值 {self.checkpoint_threshold:.3f})"
                )
        for index, score in enumerate(self.current_episode_scores, start=1):
            print(f"环境 {index} 当前未完成回合得分: {score:.3f}")

    def _checkpoint_label(self, checkpoint_number):
        if 1 <= checkpoint_number <= len(self.checkpoint_names):
            return f"{self.checkpoint_names[checkpoint_number - 1]}（检查点 {checkpoint_number}）"
        return f"检查点 {checkpoint_number}"


class EpisodeCompletePPO(PPO):
    """Train on a final partial rollout when the callback ends an episode."""

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """Collect a normal rollout, or finalize it immediately after callback stop.

        SB3 normally returns before storing the transition when a callback returns
        False. Here that transition is stored, GAE is computed over the collected
        portion, and the caller is allowed to run PPO.train() once before learning
        exits. This keeps terminal checkpoint/timeout rewards in the PPO update.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)
        n_steps = 0
        rollout_buffer.reset()
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)
        callback.on_rollout_start()
        callback_stopped = False

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()
            clipped_actions = actions
            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)
            self.num_timesteps += env.num_envs
            callback.update_locals(locals())
            continue_training = callback.on_step()

            self._update_info_buffer(infos, dones)
            n_steps += 1
            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)

            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

            if not continue_training:
                callback_stopped = True
                break

        if n_steps == 0:
            return False

        if callback_stopped:
            # RolloutBuffer normally exposes its full capacity only. Restrict this
            # update to the transitions actually collected, then restore capacity
            # after PPO.train() so the next reset allocates the usual buffer.
            rollout_buffer.buffer_size = n_steps
            rollout_buffer.full = True
            for field_name in (
                "observations",
                "actions",
                "rewards",
                "returns",
                "episode_starts",
                "values",
                "log_probs",
                "advantages",
            ):
                field = getattr(rollout_buffer, field_name)
                setattr(rollout_buffer, field_name, field[:n_steps].copy())

        with th.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        callback.update_locals(locals())
        callback.on_rollout_end()
        if callback_stopped:
            # PPO.learn() skips train() when collect_rollouts returns False, so
            # perform this finalized partial-batch update here before stopping.
            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)
            self.train()
            return False
        return True

    def train(self) -> None:
        try:
            super().train()
        finally:
            # A partial buffer temporarily changes this size for PPO's minibatches.
            if self.rollout_buffer is not None:
                self.rollout_buffer.buffer_size = self.n_steps


def create_model(env):
    """Build a CNN PPO policy for grayscale and relative-depth observations."""
    return EpisodeCompletePPO(
        "CnnPolicy",
        env,
        verbose=1,
        n_steps=128,
        batch_size=32,
        learning_rate=2.5e-4,
        device=DEVICE,
    )


def main():
    parser = argparse.ArgumentParser(description="Train a PPO agent to navigate Half-Life 1 c1a0.")
    parser.add_argument("--train", action="store_true", help="Start training and control the game window.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue training from the saved model with all config.cfg binds.",
    )
    parser.add_argument("--timesteps", type=int, default=100_000, help="Training steps when --train is set.")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override PPO's learning rate, useful when fine-tuning a behavior-cloned policy.",
    )
    parser.add_argument("--checkpoint-threshold", type=float, default=0.60)
    parser.add_argument(
        "--model-name",
        default=MODEL_PATH.name,
        help="Base filename for model saves in the models directory.",
    )
    args = parser.parse_args()
    if args.learning_rate is not None and args.learning_rate <= 0:
        parser.error("--learning-rate must be greater than zero")

    if args.train:
        start_training_log()

    model_path = MODEL_PATH.with_name(args.model_name)
    continued_model_path = model_path.with_name(f"{model_path.name}_continued")

    if args.resume and not args.train:
        parser.error("--resume requires --train")
    resume_path = (
        continued_model_path
        if continued_model_path.with_suffix(".zip").is_file()
        else model_path
    )
    if args.resume and not resume_path.with_suffix(".zip").is_file():
        parser.error(f"Cannot resume: saved model not found at {model_path}.zip")

    cuda_available = torch.cuda.is_available()
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch CUDA version: {torch.version.cuda}")
    print(f"CUDA available: {cuda_available}")
    if cuda_available:
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print(f"Requested device: {DEVICE} (CUDA is unavailable in this Python environment)")

    env = HalfLifeEnv(checkpoint_threshold=args.checkpoint_threshold)
    training_completed = False
    try:
        if args.resume:
            model = EpisodeCompletePPO.load(str(resume_path), env=env, device=DEVICE)
            print(f"Loaded PPO model from {resume_path}.zip")
        else:
            model = create_model(env)
            print("PPO CNN model initialized successfully.")
        if args.learning_rate is not None:
            model.learning_rate = args.learning_rate
            model.lr_schedule = get_schedule_fn(args.learning_rate)
            print(f"PPO learning rate override: {args.learning_rate:g}")
        if not args.train:
            print("No game input was sent. Start training with: main.py --train")
            return

        if not args.resume:
            initial_weights_dir = model_path.parent / "initial_weights"
            initial_weights_dir.mkdir(parents=True, exist_ok=True)
            snapshot_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            initial_weights_path = (
                initial_weights_dir
                / f"{model_path.name}_random_init_{snapshot_timestamp}"
            )
            model.save(str(initial_weights_path))
            print(f"Saved randomly initialized weights to {initial_weights_path}.zip")

        score_callback = TrainingScoreCallback(
            args.checkpoint_threshold,
            [folder.name for folder in env.checkpoint_folders],
        )
        model.learn(
            total_timesteps=args.timesteps,
            reset_num_timesteps=not args.resume,
            callback=score_callback,
        )
        score_callback.print_summary()
        model_path.parent.mkdir(parents=True, exist_ok=True)
        save_path = continued_model_path if args.resume else model_path
        model.save(str(save_path))
        print(f"Saved model to {save_path}.zip")
        training_completed = True
    finally:
        env.close()
        if training_completed:
            reset_game_to_c1a0()


if __name__ == "__main__":
    main()
