import argparse
import importlib.util
import os
import sys
import time

import cv2
import gymnasium as gym
import numpy as np
import pygame

from rl_snake.wrapper import ModularSnakeWrapper

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Load training config from a .py file
def load_training_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Training config file not found: {config_path}")
    spec = importlib.util.spec_from_file_location("training_config", config_path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return cfg


def record_snake_agent(
    model_path: str, output_folder: str, num_episodes: int, fps: int
):
    # Load config from the .py file
    cfg = load_training_config(model_path + ".py")

    MODEL_CLASS = cfg.MODEL_CLASS

    os.makedirs(output_folder, exist_ok=True)

    # Load model with non-rendering env (needed for SB3)
    base_env = gym.make("Snake-v1")
    env = ModularSnakeWrapper(
        base_env, state_type=cfg.STATE_TYPE, reward_type=cfg.REWARD_TYPE
    )
    model = MODEL_CLASS.load(model_path, env=env)
    env.close()

    # Rendering environment for recording
    base_env = gym.make("Snake-v1", render_mode="human")
    env = ModularSnakeWrapper(
        base_env, state_type=cfg.STATE_TYPE, reward_type=cfg.REWARD_TYPE
    )

    model_name_only = os.path.basename(model_path)

    time.sleep(0.5)  # short pause before recording

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        frames = []

        while not done:
            # Capture frame from pygame display
            screen = pygame.display.get_surface()
            if screen is not None:
                frame = pygame.surfarray.array3d(screen)
                frame = np.transpose(frame, (1, 0, 2))
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frames.append(frame)

            # Predict action
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated
            pygame.event.pump()

        # Save video
        if frames:
            output_path = os.path.join(
                output_folder,
                f"{model_name_only}_episode_{episode + 1}.mp4",
            )

            height, width = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            for frame in frames:
                out.write(frame)

            out.release()
            print(f"Saved episode {episode + 1} video: {output_path}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-path",
        type=str,
        default="./models/snake_dqn",
        help="Path to the trained model (without .zip extension)",
    )

    parser.add_argument(
        "--output-folder",
        type=str,
        default="./videos",
        help="Folder to save recorded videos",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to record",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Frames per second for the output video",
    )

    args = parser.parse_args()

    record_snake_agent(
        model_path=args.model_path,
        output_folder=args.output_folder,
        num_episodes=args.episodes,
        fps=args.fps,
    )
