import gymnasium as gym
import gymnasium_snake_game
from stable_baselines3 import DQN
import numpy as np

from rl_snake.wrapper import EnhancedSnakeWrapper


def test_snake_agent(model_path="./models/snake_agent", num_episodes=20):
    """Test the trained Snake agent"""
    print("=" * 70)
    print("Testing Snake Agent")
    print("=" * 70)

    # Create wrapped environment
    base_env = gym.make("Snake-v1")
    env = EnhancedSnakeWrapper(base_env)

    # Load model
    print(f"\nLoading model from {model_path}.zip...")
    model = DQN.load(model_path, env=env)

    results = []

    print(f"\nRunning {num_episodes} test episodes...\n")

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

            # Count food eaten
            if reward > 100:
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

    # Calculate statistics
    avg_food = np.mean([r["food"] for r in results])
    avg_length = np.mean([r["length"] for r in results])
    avg_reward = np.mean([r["reward"] for r in results])
    max_food = max([r["food"] for r in results])
    min_food = min([r["food"] for r in results])

    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS:")
    print(f"  Average Food per Episode: {avg_food:.2f}")
    print(f"  Maximum Food in Episode:  {max_food}")
    print(f"  Minimum Food in Episode:  {min_food}")
    print(f"  Average Episode Length:   {avg_length:.1f}")
    print(f"  Average Reward:           {avg_reward:.1f}")
    print("=" * 70)

    env.close()

    return results


if __name__ == "__main__":
    test_snake_agent(model_path="./models/snake_agent", num_episodes=20)
