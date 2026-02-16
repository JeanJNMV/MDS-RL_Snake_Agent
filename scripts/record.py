import gymnasium as gym
import gymnasium_snake_game
from stable_baselines3 import DQN
import numpy as np
import cv2
import os
import time
import pygame

from rl_snake.wrapper import EnhancedSnakeWrapper


def record_snake_agent(
    model_path="./models/snake_agent",
    output_folder="./videos",
    num_episodes=5,
    fps=15,
):
    """Record videos of the Snake agent playing"""
    os.makedirs(output_folder, exist_ok=True)

    # Load model with non-rendering environment first
    base_env = gym.make("Snake-v1")
    env = EnhancedSnakeWrapper(base_env)

    print(f"Loading model from {model_path}.zip...")
    model_name = os.path.basename(model_path)
    model = DQN.load(model_path, env=env)
    env.close()

    # Create rendering environment
    base_env = gym.make("Snake-v1", render_mode="human")
    env = EnhancedSnakeWrapper(base_env)

    print(f"Recording {num_episodes} episode(s) to '{output_folder}' folder...\n")

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        frames = []
        food_count = 0
        steps = 0

        time.sleep(0.5)

        while not done:
            # Capture frame
            screen = pygame.display.get_surface()
            if screen is not None:
                frame = pygame.surfarray.array3d(screen)
                frame = np.transpose(frame, (1, 0, 2))
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frames.append(frame)

            # Take action
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
                f"{model_name}_episode_{episode + 1}_food_{food_count}_steps_{steps}.mp4",
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
    print(f"\nRecording complete! Videos saved to '{output_folder}/' folder")


if __name__ == "__main__":
    record_snake_agent(
        model_path="./models/snake_agent",
        output_folder="videos",
        num_episodes=5,
        fps=15,
    )
