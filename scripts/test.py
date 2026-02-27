import argparse
import os
import sys

import gymnasium as gym
import numpy as np

from rl_snake.wrapper import ModularSnakeWrapper
from rl_snake.utils import load_training_config

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Testing
def test_snake_agent(model_path: str, num_episodes: int):
    cfg = load_training_config(model_path + ".py")
    MODEL_CLASS = cfg.MODEL_CLASS

    print(f"\nTesting {MODEL_CLASS.__name__} Agent")
    print(f"State: {cfg.STATE_TYPE} | Reward: {cfg.REWARD_TYPE}")

    # Create environment matching training
    base_env = gym.make("Snake-v1")
    env = ModularSnakeWrapper(
        base_env,
        state_type=cfg.STATE_TYPE,
        reward_type=cfg.REWARD_TYPE,
    )

    print(f"\nLoading model from {model_path}.zip")
    model = MODEL_CLASS.load(model_path, env=env)

    results = []

    print(f"\nRunning {num_episodes} test episodes.\n")

    for episode in range(num_episodes):
        obs, info = env.reset()

        terminated = False
        truncated = False

        ep_reward = 0.0
        ep_length = 0
        food_count = 0

        while not (terminated or truncated):
            # Store previous food position BEFORE stepping
            previous_food = info.get("food")

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            ep_reward += reward
            ep_length += 1

            # Food counting
            current_head = info.get("head")

            if previous_food is not None and current_head == previous_food:
                food_count += 1

        results.append(
            {
                "episode": episode + 1,
                "food": food_count,
                "length": ep_length,
                "reward": ep_reward,
            }
        )

        print(
            f"Episode {episode + 1:2d}: "
            f"Food={food_count:3d} | "
            f"Steps={ep_length:4d} | "
            f"Reward={ep_reward:8.1f}"
        )

    # Statistics
    avg_food = np.mean([r["food"] for r in results])
    avg_length = np.mean([r["length"] for r in results])
    avg_reward = np.mean([r["reward"] for r in results])
    max_food = max([r["food"] for r in results])
    min_food = min([r["food"] for r in results])

    print("\nFINAL TEST RESULTS:")
    print(f"  Average Food per Episode: {avg_food:.2f}")
    print(f"  Maximum Food in Episode:  {max_food}")
    print(f"  Minimum Food in Episode:  {min_food}")
    print(f"  Average Episode Length:   {avg_length:.1f}")
    print(f"  Average Reward:           {avg_reward:.1f}")

    env.close()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path", type=str, default="./models/dqn_full_grid_dense"
    )
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()

    test_snake_agent(model_path=args.model_path, num_episodes=args.episodes)
