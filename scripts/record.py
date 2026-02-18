import argparse
import os
import sys
import time

import cv2
import gymnasium as gym
import numpy as np
import pygame

from rl_snake.wrapper import EnhancedSnakeWrapper
from rl_snake.utils import load_model_class

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Recording
def record_snake_agent(
    model_name: str,
    model_path: str,
    output_folder: str,
    num_episodes: int,
    fps: int,
):
    os.makedirs(output_folder, exist_ok=True)

    MODEL_CLASS = load_model_class(model_name)

    # Load model with non-rendering env
    base_env = gym.make("Snake-v1")
    env = EnhancedSnakeWrapper(base_env)

    print(f"Loading model from {model_path}.zip.")
    model = MODEL_CLASS.load(model_path, env=env)
    env.close()

    # Rendering env
    base_env = gym.make("Snake-v1", render_mode="human")
    env = EnhancedSnakeWrapper(base_env)

    model_name_only = os.path.basename(model_path)

    print(f"Recording {num_episodes} episode(s) to '{output_folder}' folder.\n")

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        frames = []
        food_count = 0
        steps = 0

        time.sleep(0.5)

        while not done:
            screen = pygame.display.get_surface()
            if screen is not None:
                frame = pygame.surfarray.array3d(screen)
                frame = np.transpose(frame, (1, 0, 2))
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frames.append(frame)

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            if reward > 100:
                food_count += 1

            steps += 1
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

            print(f"Episode {episode + 1}: Saved {output_path}")
            print(
                f"  Food eaten: {food_count} | Steps: {steps} | Frames: {len(frames)}"
            )
        else:
            print(f"Episode {episode + 1}: No frames captured")

    env.close()
    print(f"\nRecording complete! Videos saved to '{output_folder}/'")


# -------------------------------------------------
# CLI
# -------------------------------------------------
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

    args = parser.parse_args()

    model_name = args.model
    model_path = args.model_path if args.model_path else f"./models/snake_{model_name}"

    record_snake_agent(
        model_name=model_name,
        model_path=model_path,
        output_folder=args.output_folder,
        num_episodes=args.episodes,
        fps=args.fps,
    )
