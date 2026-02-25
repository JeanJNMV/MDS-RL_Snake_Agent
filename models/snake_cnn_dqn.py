# Auto-generated training configuration
from stable_baselines3 import DQN

MODEL_CLASS = DQN
IS_CNN = True

STATE_TYPE = 'cnn_full_grid'
REWARD_TYPE = 'dense'
USE_VEC_ENV = False
N_ENVS = 4

# Original MODEL_CONFIG hyperparameters:
# policy: CnnPolicy
# policy_kwargs: {'features_extractor_class': <class 'rl_snake.cnn.CustomSnakeCNN'>, 'features_extractor_kwargs': {'features_dim': 256}}
# learning_rate: 0.0001
# buffer_size: 100000
# learning_starts: 10000
# batch_size: 32
# gamma: 0.99
# exploration_fraction: 0.5
# exploration_initial_eps: 1.0
# exploration_final_eps: 0.05
# verbose: 0
