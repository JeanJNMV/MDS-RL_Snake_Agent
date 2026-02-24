from collections import deque


class BaseReward:
    """Base class for reward functions in the Snake RL environment."""

    def reset(self):
        pass

    def compute(self, reward, terminated, truncated, info, previous_info, steps):
        raise NotImplementedError


# Sparse Reward
class SparseReward(BaseReward):
    """
    SparseReward class for computing sparse reward signals in reinforcement learning.

    This reward structure provides minimal feedback to the agent:
    - Positive reward when food is consumed
    - Penalty when the agent dies or episode times out
    - No intermediate rewards for other actions

    Attributes:
        food_reward (float): Reward value for consuming food. Default is 1.0
        death_penalty (float): Penalty value for death or timeout. Default is -1.0

    Methods:
        compute(reward, terminated, truncated, info, previous_info, steps): Calculate the total reward based on game events.
    """

    def __init__(self, food_reward=1.0, death_penalty=-1.0):
        self.food_reward = food_reward
        self.death_penalty = death_penalty

    def compute(self, reward, terminated, truncated, info, previous_info, steps):
        total_reward = 0.0

        # --- Food ---
        if reward > 0:
            total_reward += self.food_reward

        # --- Death ---
        if terminated:
            total_reward += self.death_penalty

        # --- Timeout ---
        if truncated:
            total_reward += self.death_penalty

        return total_reward


# Dense Reward
class DenseReward(BaseReward):
    """
    DenseReward class for providing dense reward shaping in RL Snake agent.

    This reward structure provides detailed feedback to guide learning:
    - Positive reward when food is consumed
    - Penalty when the agent dies or episode times out
    - Distance-based shaping rewards for moving toward/away from food
    - Loop detection penalty for repetitive movements

    Attributes:
        food_reward (float): Reward value for consuming food. Default is 10.0
        death_penalty (float): Penalty value for death or timeout. Default is -10.0
        distance_scale (float): Scaling factor for distance-based rewards. Default is 1.0
        loop_penalty (float): Penalty value for detected loops. Default is -5.0
        timeout_penalty (float): Penalty value for episode timeouts. Default is -5.0
        loop_window (int): Size of position history window for loop detection. Default is 20
        loop_threshold (int): Number of repeated positions before triggering penalty. Default is 4

    Methods:
        reset(): Clear internal state between episodes.
        compute(reward, terminated, truncated, info, previous_info, steps): Calculate the total reward based on game events.
    """

    def __init__(
        self,
        food_reward=10.0,
        death_penalty=-10.0,
        distance_scale=1.0,
        loop_penalty=-5.0,
        timeout_penalty=-5.0,
        loop_window=20,
        loop_threshold=4,
    ):
        self.food_reward = food_reward
        self.death_penalty = death_penalty
        self.distance_scale = distance_scale
        self.loop_penalty = loop_penalty
        self.timeout_penalty = timeout_penalty

        self.loop_threshold = loop_threshold
        self.position_history = deque(maxlen=loop_window)

        self.previous_distance = None

    def reset(self):
        self.previous_distance = None
        self.position_history.clear()

    def compute(self, reward, terminated, truncated, info, previous_info, steps):
        total_reward = 0.0

        head = info.get("head")
        food = info.get("food")

        # --- Food ---
        if reward > 0:
            total_reward += self.food_reward
            self.previous_distance = None

        # --- Death ---
        if terminated:
            total_reward += self.death_penalty

        # --- Timeout ---
        if truncated:
            total_reward += self.timeout_penalty

        # --- Loop detection ---
        if head:
            self.position_history.append(head)
            if self.position_history.count(head) >= self.loop_threshold:
                total_reward += self.loop_penalty

        # --- Distance shaping ---
        if head and food:
            current_distance = abs(head[0] - food[0]) + abs(head[1] - food[1])

            if self.previous_distance is not None:
                delta = self.previous_distance - current_distance
                total_reward += delta * self.distance_scale

            self.previous_distance = current_distance

        return total_reward


# Factory
def get_reward_function(name: str):
    if name == "sparse":
        return SparseReward()
    elif name == "dense":
        return DenseReward()
    else:
        raise ValueError(f"Unknown reward type: {name}")
