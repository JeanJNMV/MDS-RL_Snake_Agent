from stable_baselines3 import DQN

MODEL_CLASS = DQN

MODEL_CONFIG = dict(
    policy="MlpPolicy",
    learning_rate=1e-4,
    buffer_size=100000,
    learning_starts=10000,
    batch_size=64,
    gamma=0.99,
    exploration_fraction=0.4,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.02,
    policy_kwargs=dict(
        net_arch=[256, 256],
    ),
    verbose=0,
)
