import argparse
from pathlib import Path

from stable_baselines3 import PPO

from hl1_env import HalfLifeEnv


MODEL_PATH = Path(__file__).resolve().parent / "models" / "c1a0_ppo_temporal_stuck_aware"
CONTINUED_MODEL_PATH = MODEL_PATH.with_name(f"{MODEL_PATH.name}_continued")


def create_model(env):
    """Build the PPO CNN policy for the environment's grayscale screenshots."""
    return PPO(
        "CnnPolicy",
        env,
        verbose=1,
        n_steps=128,
        batch_size=32,
        learning_rate=2.5e-4,
    )


def main():
    parser = argparse.ArgumentParser(description="Train a PPO agent to navigate Half-Life 1 c1a0.")
    parser.add_argument("--train", action="store_true", help="Start training and control the game window.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue training from the saved temporal stuck-aware model.",
    )
    parser.add_argument("--timesteps", type=int, default=100_000, help="Training steps when --train is set.")
    parser.add_argument("--checkpoint-threshold", type=float, default=0.75)
    args = parser.parse_args()

    if args.resume and not args.train:
        parser.error("--resume requires --train")
    resume_path = (
        CONTINUED_MODEL_PATH
        if CONTINUED_MODEL_PATH.with_suffix(".zip").is_file()
        else MODEL_PATH
    )
    if args.resume and not resume_path.with_suffix(".zip").is_file():
        parser.error(f"Cannot resume: saved model not found at {MODEL_PATH}.zip")

    env = HalfLifeEnv(checkpoint_threshold=args.checkpoint_threshold)
    try:
        if args.resume:
            model = PPO.load(str(resume_path), env=env)
            print(f"Loaded PPO model from {resume_path}.zip")
        else:
            model = create_model(env)
            print("PPO CNN model initialized successfully.")
        if not args.train:
            print("No game input was sent. Start training with: main.py --train")
            return

        model.learn(total_timesteps=args.timesteps, reset_num_timesteps=not args.resume)
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        save_path = CONTINUED_MODEL_PATH if args.resume else MODEL_PATH
        model.save(str(save_path))
        print(f"Saved model to {save_path}.zip")
    finally:
        env.close()


if __name__ == "__main__":
    main()
