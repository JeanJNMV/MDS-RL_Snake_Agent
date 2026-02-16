import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN

from rl_snake.wrapper import EnhancedSnakeWrapper, TrainingMonitor


def train_snake_agent(model_path="./models/snake_agent", timesteps=500_000):
    """Train DQN agent on Snake"""
    print("Initializing Snake environment...")

    base_env = gym.make("Snake-v1")
    env = EnhancedSnakeWrapper(base_env)

    callback = TrainingMonitor(log_interval=100)

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=100000,
        learning_starts=10000,
        batch_size=32,
        gamma=0.99,
        exploration_fraction=0.5,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        verbose=0,
    )

    print(f"Training for {timesteps:,} steps...\n")
    model.learn(total_timesteps=timesteps, callback=callback, progress_bar=True)

    model.save(model_path)

    print("\nTraining complete!")
    print(f"Model saved as '{model_path}.zip'")
    print(f"Total food eaten: {callback.food_count}")
    print(f"Final avg reward: {np.mean(callback.episode_rewards[-100:]):.2f}")

    env.close()


if __name__ == "__main__":
    train_snake_agent(model_path="./models/snake_agent", timesteps=1_000_000)
