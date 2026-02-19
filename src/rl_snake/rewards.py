class BaseReward:
    def reset(self):
        pass

    def compute(self, reward, terminated, truncated, info, previous_info, steps):
        raise NotImplementedError


# Sparse Reward
class SparseReward(BaseReward):
    def compute(self, reward, terminated, truncated, info, previous_info, steps):
        if reward > 0:  # food eaten
            return 1.0
        if terminated:  # died
            return -1.0
        return 0.0


# Dense Reward
class DenseReward(BaseReward):
    def __init__(self):
        self.previous_distance = None
        self.position_history = []

    def reset(self):
        self.previous_distance = None
        self.position_history = []

    def compute(self, reward, terminated, truncated, info, previous_info, steps):
        head = info.get("head")
        food = info.get("food")

        # Loop detection
        if head:
            self.position_history.append(head)

            if len(self.position_history) > 20:
                self.position_history.pop(0)

            if self.position_history.count(head) >= 4:
                return -50.0

        # Timeout penalty
        if steps > 1000:
            return -20.0

        # Reward shaping
        if reward > 0:
            self.previous_distance = None
            return 1000.0

        if terminated:
            return -10.0

        if head and food:
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
            return shaped_reward

        return -0.1


# Factory
def get_reward_function(name: str):
    if name == "sparse":
        return SparseReward()
    elif name == "dense":
        return DenseReward()
    else:
        raise ValueError(f"Unknown reward type: {name}")
