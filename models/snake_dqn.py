# Auto-generated training configuration
from stable_baselines3 import DQN

MODEL_CLASS = DQN

MODEL_CONFIG = {
    "policy": "MlpPolicy",
    "learning_rate": 0.0001,
    "buffer_size": 100000,
    "learning_starts": 10000,
    "batch_size": 32,
    "gamma": 0.99,
    "exploration_fraction": 0.5,
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.05,
    "verbose": 0
}
STATE_TYPE = 'full_grid'
REWARD_TYPE = 'dense'
USE_VEC_ENV = False
N_ENVS = 4
