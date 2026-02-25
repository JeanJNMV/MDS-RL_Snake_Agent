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
    """
    EgocentricEncoder encodes a localized window around the agent's head position.

    This encoder extracts a window-based egocentric view centered around the agent,
    converting it into a flattened one-dimensional array suitable for neural network input.

    Attributes:
        window_radius (int): The radius of the observation window. Default is 3,
            creating a 7x7 window.
        observation_space (gym.spaces.Box): The observation space of shape (147,)
            with values normalized between 0 and 1 as float32.

    Methods:
        encode(obs, info): Extracts and flattens the egocentric window.
    """

    def __init__(self):
        super().__init__()
        self.window_radius = 3
        window_size = self.window_radius * 2 + 1
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(window_size * window_size * 3,),
            dtype=np.float32,
        )

    def encode(self, obs, info):
        head = info.get("head", (obs.shape[0] // 2, obs.shape[1] // 2))
        r = np.clip(head[0], 0, obs.shape[0] - 1)
        c = np.clip(head[1], 0, obs.shape[1] - 1)

        pad_width = (
            (self.window_radius, self.window_radius),
            (self.window_radius, self.window_radius),
            (0, 0),
        )
        padded_obs = np.pad(obs, pad_width, mode="constant", constant_values=0)
        window_size = self.window_radius * 2 + 1
        cropped_obs = padded_obs[r : r + window_size, c : c + window_size, :]
        return cropped_obs.flatten().astype(np.float32)


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
    Convolutional Neural Network Grid Encoder for Snake Game State.

    This encoder maintains the spatial structure of the game grid as a 3D tensor,
    making it suitable for CNN-based agents that benefit from spatial feature extraction.

    Attributes:
        observation_space (gym.spaces.Box): A 16x16 grid with 3 channels (RGB),
            with values normalized to [0, 1] as float32.

    Methods:
        encode(obs, info): Converts raw observations to CNN-compatible format.
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


class CnnEgocentricEncoder(BaseStateEncoder):
    """
    Convolutional Neural Network Egocentric Encoder for Snake Game State.

    This encoder extracts a localized egocentric view around the agent's head position,
    maintaining the spatial structure for CNN-based agents.

    Attributes:
        window_radius (int): The radius of the observation window. Default is 3,
            creating a 7x7 window.
        observation_space (gym.spaces.Box): A 7x7 grid with 3 channels (RGB),
            with values normalized to [0, 1] as float32.
    Methods:
        encode(obs, info): Extracts and formats the egocentric view for CNN input.
    """

    def __init__(self):
        super().__init__()
        self.window_radius = 3
        self.window_size = self.window_radius * 2 + 1
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(self.window_size, self.window_size, 3),
            dtype=np.float32,
        )

    def encode(self, obs, info):
        head = info.get("head", (obs.shape[0] // 2, obs.shape[1] // 2))
        r = np.clip(head[0], 0, obs.shape[0] - 1)
        c = np.clip(head[1], 0, obs.shape[1] - 1)

        pad_width = (
            (self.window_radius, self.window_radius),
            (self.window_radius, self.window_radius),
            (0, 0),
        )
        padded_obs = np.pad(obs, pad_width, mode="constant", constant_values=0)
        cropped_obs = padded_obs[r : r + self.window_size, c : c + self.window_size, :]
        return cropped_obs.astype(np.float32)


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
    elif name == "cnn_egocentric":
        return CnnEgocentricEncoder()
    else:
        raise ValueError(f"Unknown state type: {name}")
