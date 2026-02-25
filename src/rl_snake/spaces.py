import gymnasium as gym
import numpy as np


# Base class
class BaseStateEncoder:
    """Base class for state encoders in the Snake RL environment."""

    def __init__(self):
        self.observation_space = None

    def encode(self, obs, info):
        raise NotImplementedError


# Full grid representation
class FullGridEncoder(BaseStateEncoder):
    """
    FullGridEncoder encodes the full game grid as a flattened observation vector.

    This encoder converts the raw observation (a 16x16x3 grid) into a flattened
    one-dimensional array of floating-point values suitable for neural network input.

    Attributes:
        observation_space (gym.spaces.Box): The observation space of shape (768,)
            with values normalized between 0 and 1 as float32.

    Methods:
        encode(obs, info): Flattens and converts the observation to float32 format.
    """

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


class CnnGridEncoder(BaseStateEncoder):
    """
    Keeps the 16x16x3 spatial grid intact for CNN processing.
    """

    def __init__(self):
        super().__init__()
        # Keep the 3D shape: (Height, Width, Channels)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(16, 16, 3),
            dtype=np.float32,
        )

    def encode(self, obs, info):
        # No flattening
        return obs.astype(np.float32)


# Factory
def get_state_encoder(name: str):
    if name == "full_grid":
        return FullGridEncoder()
    elif name == "egocentric":
        return EgocentricEncoder()
    elif name == "features":
        return FeatureVectorEncoder()
    elif name == "cnn_full_grid":
        return CnnGridEncoder()
    else:
        raise ValueError(f"Unknown state type: {name}")
