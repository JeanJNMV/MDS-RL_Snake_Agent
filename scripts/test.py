import argparse
import os
import sys

import gymnasium as gym
import numpy as np

from rl_snake.wrapper import ModularSnakeWrapper
from rl_snake.utils import load_model_class

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Testing
def test_snake_agent(
    model_name: str,
    model_path: str,
    num_episodes: int,
    state_type: str,
    reward_type: str,
):
    print(f"Testing {model_name.upper()} Agent")
    print(f"State: {state_type} | Reward: {reward_type}")

    MODEL_CLASS = load_model_class(model_name)

    # Environment (must match training setup)
    base_env = gym.make("Snake-v1")
    env = ModularSnakeWrapper(
        base_env,
        state_type=state_type,
        reward_type=reward_type,
    )

    print(f"\nLoading model from {model_path}.zip")
    model = MODEL_CLASS.load(model_path, env=env)

    results = []

    print(f"\nRunning {num_episodes} test episodes.\n")

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0
        ep_length = 0
        food_count = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            ep_reward += reward
            ep_length += 1

            # Food detection should use original env signal
            if info.get("food_eaten", False):
                food_count += 1

            # If your base env doesn't provide "food_eaten",
            # fallback to raw reward signal:
            if reward_type == "dense" and reward == 1000.0:
                food_count += 1
            elif reward_type == "sparse" and reward == 1.0:
                food_count += 1

            done = terminated or truncated

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
        "--episodes",
        type=int,
        default=20,
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

    test_snake_agent(
        model_name=model_name,
        model_path=model_path,
        num_episodes=args.episodes,
        state_type=args.state,
        reward_type=args.reward,
    )
