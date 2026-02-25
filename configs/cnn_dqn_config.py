from stable_baselines3 import DQN
from rl_snake.cnn import CustomSnakeCNN

MODEL_CLASS = DQN

MODEL_CONFIG = dict(
    policy="CnnPolicy",  # Changed from MlpPolicy
    policy_kwargs=dict(
        features_extractor_class=CustomSnakeCNN,
        features_extractor_kwargs=dict(features_dim=256),
    ),
    learning_rate=1e-4,
    buffer_size=100000,
    learning_starts=10000,
    batch_size=32,
    gamma=0.99,
    exploration_fraction=0.5,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    verbose=0,
)
