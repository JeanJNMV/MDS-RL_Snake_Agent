import argparse
import os
import sys

import gymnasium as gym
import gymnasium_snake_game
import numpy as np

from stable_baselines3.common.env_util import make_vec_env

from rl_snake.wrapper import EnhancedSnakeWrapper, TrainingMonitor
from rl_snake.utils import load_config

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Environment Factory
def make_env():
    base_env = gym.make("Snake-v1")
    return EnhancedSnakeWrapper(base_env)


# Training
def train(
    model_name: str,
    timesteps: int,
    save_path: str,
    use_vec_env: bool,
    tensorboard_log: str,
    n_envs: int,
):
    print(f"\nTraining {model_name.upper()}\n")

    MODEL_CLASS, MODEL_CONFIG = load_config(model_name)

    # Environment creation (using vectorized env for on-policy algorithms)
    if model_name in ["a2c", "ppo"] and use_vec_env:
        print(f"Using vectorized environment ({n_envs} envs)")
        env = make_vec_env(make_env, n_envs=n_envs)
    else:
        print("Using single environment")
        env = make_env()

    callback = TrainingMonitor(log_interval=100)

    # Create model
    model = MODEL_CLASS(
        env=env,
        tensorboard_log=tensorboard_log,
        **MODEL_CONFIG,
    )

    print(f"TensorBoard logs: {tensorboard_log}")
    print(f"Training for {timesteps:,} steps\n")

    model.learn(
        total_timesteps=timesteps,
        callback=callback,
        progress_bar=True,
    )

    model.save(save_path)

    print("\nTraining complete!")
    print(f"Model saved as '{save_path}.zip'")

    if len(callback.episode_rewards) > 0:
        print(f"Total food eaten: {callback.food_count}")
        print(
            f"Final avg reward (last 100): "
            f"{np.mean(callback.episode_rewards[-100:]):.2f}"
        )

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        choices=["dqn", "double_dqn", "a2c", "ppo"],
        default="dqn",
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=1_000_000,
    )

    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--tensorboard-log",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--use-vec-env",
        action="store_true",
        default=True,
        help="Use vectorized environment (for A2C/PPO). Default: True",
    )

    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Number of environments for vectorized training",
    )

    args = parser.parse_args()

    # Default Paths
    model_name = args.model

    save_path = args.save_path if args.save_path else f"./models/snake_{model_name}"

    tensorboard_log = (
        args.tensorboard_log if args.tensorboard_log else f"./logs/{model_name}/"
    )

    os.makedirs("./models", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)

    train(
        model_name=model_name,
        timesteps=args.timesteps,
        save_path=save_path,
        use_vec_env=args.use_vec_env,
        tensorboard_log=tensorboard_log,
        n_envs=args.n_envs,
    )
