import torch as th
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomSnakeCNN(BaseFeaturesExtractor):
    """
    CustomSnakeCNN - A custom CNN feature extractor for Snake game observations.

    This class extracts features from game observations using a convolutional neural network
    followed by fully connected layers. It is designed to work with Stable Baselines3 (SB3)
    and automatically handles the conversion of observation shapes from (Height, Width, Channels)
    to (Channels, Height, Width) format required by PyTorch.

    Attributes:
        cnn (nn.Sequential): Sequential module containing convolutional layers and activation functions.
        linear (nn.Sequential): Sequential module containing a linear layer and ReLU activation.

    Methods:
        __init__(observation_space, features_dim):
            Initializes the CustomSnakeCNN with convolutional and linear layers.

        forward(observations):
            Performs a forward pass through the CNN and linear layers.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        # SB3 automatically changes (Height, Width, Channels) to (Channels, Height, Width) for PyTorch
        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Automatically calculate the output shape of the CNN
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))
