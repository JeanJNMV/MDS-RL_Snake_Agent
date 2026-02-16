import gymnasium as gym
import numpy as np

from gymnasium import ObservationWrapper, Wrapper
from stable_baselines3.common.callbacks import BaseCallback


class EnhancedSnakeWrapper(ObservationWrapper, Wrapper):
    """Combined wrapper: enhanced observations + reward shaping + loop detection"""

    def __init__(self, env):
        super().__init__(env)

        # Observation space setup
        grid_size = 16 * 16 * 3
        extra_features = 6
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(grid_size + extra_features,), dtype=np.float32
        )

        # State tracking
        self.last_info = {}
        self.previous_distance = None
        self.position_history = []
        self.steps_in_episode = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_info = info
        self.previous_distance = None
        self.position_history = []
        self.steps_in_episode = 0

        head = info.get("head")
        food = info.get("food")
        if head and food:
            self.previous_distance = abs(head[0] - food[0]) + abs(head[1] - food[1])

        return self.observation(obs), info

    def observation(self, obs):
        flat_grid = obs.flatten()
        head = self.last_info.get("head", (8, 8))
        food = self.last_info.get("food", (8, 8))

        head_x = head[0] / 16.0
        head_y = head[1] / 16.0
        food_x = food[0] / 16.0
        food_y = food[1] / 16.0
        dx = (food[0] - head[0]) / 16.0
        dy = (food[1] - head[1]) / 16.0

        enhanced_obs = np.concatenate(
            [flat_grid, [head_x, head_y, food_x, food_y, dx, dy]]
        ).astype(np.float32)

        return enhanced_obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.last_info = info
        self.steps_in_episode += 1

        head = info.get("head")
        food = info.get("food")

        # Detect infinite loops: if snake visits same position too many times
        if head:
            self.position_history.append(head)

            # Keep only last 20 positions
            if len(self.position_history) > 20:
                self.position_history.pop(0)

            # Check for loops: if position appears 4+ times in last 20 steps
            if self.position_history.count(head) >= 4:
                terminated = True
                shaped_reward = -50.0
                return self.observation(obs), shaped_reward, terminated, truncated, info

        # Timeout: force end if episode is too long without progress
        if self.steps_in_episode > 1000:
            truncated = True
            shaped_reward = -20.0
            return self.observation(obs), shaped_reward, terminated, truncated, info

        # Reward shaping
        shaped_reward = 0

        if reward > 0:  # Food eaten
            shaped_reward = 1000.0
        elif terminated:  # Died
            shaped_reward = -10.0
        elif head and food:
            current_distance = abs(head[0] - food[0]) + abs(head[1] - food[1])

            if self.previous_distance is not None:
                distance_change = self.previous_distance - current_distance

                if distance_change > 0:
                    shaped_reward = 1.0
                elif distance_change < 0:
                    shaped_reward = -1.0
                else:
                    shaped_reward = -0.1
            else:
                shaped_reward = -0.1

            self.previous_distance = current_distance
        else:
            shaped_reward = -0.1

        return self.observation(obs), shaped_reward, terminated, truncated, info


class TrainingMonitor(BaseCallback):
    """Simple training progress monitor"""

    def __init__(self, log_interval=100):
        super().__init__()
        self.log_interval = log_interval
        self.episode_rewards = []
        self.episode_lengths = []
        self.food_count = 0

    def _on_step(self):
        reward = self.locals.get("rewards", [0])[0]

        if reward > 100:
            self.food_count += 1

        if self.locals.get("dones", [False])[0]:
            info = self.locals["infos"][0]
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])

                if len(self.episode_rewards) % self.log_interval == 0:
                    recent_rewards = self.episode_rewards[-self.log_interval :]
                    recent_lengths = self.episode_lengths[-self.log_interval :]

                    print(
                        f"Episodes: {len(self.episode_rewards):5d} | "
                        f"Avg Reward: {np.mean(recent_rewards):7.2f} | "
                        f"Avg Length: {np.mean(recent_lengths):6.1f} | "
                        f"Total Food: {self.food_count:5d}"
                    )

        return True
