import numpy as np


class BaseAgent:
    def __init__(self, path):
        pass

    def choose_action(self, state):
        raise NotImplementedError

    def update(self, state, action, reward, next_state, done):
        raise NotImplementedError


class RandomAgent(BaseAgent):
    def __init__(self):
        pass

    def choose_action(self, state):
        return np.random.randint(0, 4)

    def update(self, state, action, reward, next_state, done):
        pass
