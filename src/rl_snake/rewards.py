from collections import deque


class BaseReward:
    def reset(self):
        pass

    def compute(self, reward, terminated, truncated, info, previous_info, steps):
        raise NotImplementedError


# Sparse Reward
class SparseReward(BaseReward):
    def __init__(self, food_reward=1.0, death_penalty=-1.0):
        self.food_reward = food_reward
        self.death_penalty = death_penalty

    def compute(self, reward, terminated, truncated, info, previous_info, steps):
        total_reward = 0.0

        if reward > 0:
            total_reward += self.food_reward

        if terminated:
            total_reward += self.death_penalty

        if truncated:
            total_reward += self.death_penalty

        return total_reward


# Dense Reward
class DenseReward(BaseReward):
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
