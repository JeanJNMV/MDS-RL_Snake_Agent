import argparse
import os
import sys
import time

import cv2
import gymnasium as gym
import numpy as np
import pygame

from rl_snake.wrapper import ModularSnakeWrapper
from rl_snake.utils import load_model_class

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def record_snake_agent(
    model_name: str,
    model_path: str,
    output_folder: str,
    num_episodes: int,
    fps: int,
    state_type: str,
    reward_type: str,
):
    os.makedirs(output_folder, exist_ok=True)

    MODEL_CLASS = load_model_class(model_name)

    # Load model with non-rendering env (for SB3)
    base_env = gym.make("Snake-v1")
    env = ModularSnakeWrapper(base_env, state_type=state_type, reward_type=reward_type)
    model = MODEL_CLASS.load(model_path, env=env)
    env.close()

    # Rendering env
    base_env = gym.make("Snake-v1", render_mode="human")
    env = ModularSnakeWrapper(base_env, state_type=state_type, reward_type=reward_type)

    model_name_only = os.path.basename(model_path)

    time.sleep(0.5)  # short pause before recording

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        frames = []

        while not done:
            # Capture frame
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
        "--model-path",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--output-folder",
        type=str,
        default="./videos",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--state",
        type=str,
        choices=["full_grid", "egocentric", "features"],
        default="full_grid",
    )

    parser.add_argument(
        "--reward",
        type=str,
        choices=["sparse", "dense"],
        default="dense",
    )

    args = parser.parse_args()

    model_name = args.model
    model_path = args.model_path if args.model_path else f"./models/snake_{model_name}"

    record_snake_agent(
        model_name=model_name,
        model_path=model_path,
        output_folder=args.output_folder,
        num_episodes=args.episodes,
        fps=args.fps,
        state_type=args.state,
        reward_type=args.reward,
    )
