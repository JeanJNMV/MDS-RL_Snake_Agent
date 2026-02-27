import argparse
import os
import sys

import gymnasium as gym
import numpy as np

from stable_baselines3.common.env_util import make_vec_env

from rl_snake.wrapper import ModularSnakeWrapper
from rl_snake.utils import load_config, TrainingMonitor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Environment Factory
def make_env(state_type, reward_type):
    base_env = gym.make("Snake-v1")
    return ModularSnakeWrapper(
        base_env,
        state_type=state_type,
        reward_type=reward_type,
    )


# Save configuration alongside model
def save_training_config(
    save_path, model_class, model_config, state_type, reward_type, use_vec_env, n_envs
):
    py_path = save_path + ".py"

    # Check if this model is using the CNN
    is_cnn = model_config.get("policy") == "CnnPolicy"

    with open(py_path, "w") as f:
        f.write("# Auto-generated training configuration\n")
        f.write(f"from stable_baselines3 import {model_class.__name__}\n\n")

        f.write(f"MODEL_CLASS = {model_class.__name__}\n")
        f.write(f"IS_CNN = {is_cnn}\n\n")  # Add a clean boolean flag

        f.write(f"STATE_TYPE = '{state_type}'\n")
        f.write(f"REWARD_TYPE = '{reward_type}'\n")
        f.write(f"USE_VEC_ENV = {use_vec_env}\n")
        f.write(f"N_ENVS = {n_envs}\n\n")

        # Save the config as comments for human readability only
        f.write("# Original MODEL_CONFIG hyperparameters:\n")
        for key, value in model_config.items():
            f.write(f"# {key}: {value}\n")

    print(f"Training configuration saved as '{py_path}'")


# Training
def train(
    model_name: str,
    timesteps: int,
    save_path: str,
    use_vec_env: bool,
    tensorboard_log: str,
    n_envs: int,
    state_type: str,
    reward_type: str,
):
    print(f"\nTraining {model_name.upper()}\n")

    MODEL_CLASS, MODEL_CONFIG = load_config(model_name)

    # Environment creation (using vectorized env for on-policy algorithms)
    if model_name in ["a2c", "ppo"] and use_vec_env:
        print(f"Using vectorized environment ({n_envs} envs)")
        env = make_vec_env(lambda: make_env(state_type, reward_type), n_envs=n_envs)
    else:
        print("Using single environment")
        env = make_env(state_type, reward_type)

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
    save_training_config(
        save_path,
        MODEL_CLASS,
        MODEL_CONFIG,
        state_type,
        reward_type,
        use_vec_env,
        n_envs,
    )

    print("\nTraining complete!")
    print(f"Model saved as '{save_path}.zip'")

    if len(callback.episode_rewards) > 0:
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
        choices=["dqn", "double_dqn", "a2c", "ppo", "cnn_dqn"],
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
        "--state",
        type=str,
        choices=[
            "full_grid",
            "egocentric",
            "features",
        ],
        default="full_grid",
    )

    parser.add_argument(
        "--reward",
        type=str,
        choices=["sparse", "dense"],
        default="dense",
    )

    parser.add_argument(
        "--tensorboard-log",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--use-vec-env",
        action="store_true",
        default=False,
        help="Use vectorized environment (for A2C/PPO). Default: False",
    )

    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Number of environments for vectorized training (only applies if --use-vec-env is set). Default: 4",
    )

    args = parser.parse_args()

    # Default Paths
    model_name = args.model

    save_path = (
        args.save_path
        if args.save_path
        else f"./models/{model_name}_{args.state}_{args.reward}"
    )

    tensorboard_log = (
        args.tensorboard_log if args.tensorboard_log else f"./logs/{model_name}/"
    )

    # Automatically adjust state type for CNN-based models
    if "cnn" in model_name:
        if args.state == "features":
            raise ValueError("CNN-based models cannot use 'features' state type.")
        if "cnn" not in args.state:
            args.state = "cnn_" + args.state

    os.makedirs("./models", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)

    train(
        model_name=model_name,
        timesteps=args.timesteps,
        save_path=save_path,
        use_vec_env=args.use_vec_env,
        tensorboard_log=tensorboard_log,
        n_envs=args.n_envs,
        state_type=args.state,
        reward_type=args.reward,
    )
