import gymnasium as gym
import numpy as np


# Base class
class BaseStateEncoder:
    def __init__(self):
        self.observation_space = None

    def encode(self, obs, info):
        raise NotImplementedError


# Full grid representation
class FullGridEncoder(BaseStateEncoder):
    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(16 * 16 * 3,),
            dtype=np.float32,
        )

    def encode(self, obs, info):
        return obs.flatten().astype(np.float32)


# Egocentric representation
class EgocentricEncoder(BaseStateEncoder):
    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(7 * 7 * 3,),  # THIS IS JUST AN EXAMPLE, TO BE ADJUSTED
            dtype=np.float32,
        )

    def encode(self, obs, info):
        pass  # Implement cropping around the snake's head and flattening


# Hand-crafted features
class FeatureVectorEncoder(BaseStateEncoder):
    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(10,),  # example feature size
            dtype=np.float32,
        )

    def encode(self, obs, info):
        pass  # Implement feature extraction (e.g., distance to food, direction to food, etc.)


# Factory
def get_state_encoder(name: str):
    if name == "full_grid":
        return FullGridEncoder()
    elif name == "egocentric":
        return EgocentricEncoder()
    elif name == "features":
        return FeatureVectorEncoder()
    else:
        raise ValueError(f"Unknown state type: {name}")
