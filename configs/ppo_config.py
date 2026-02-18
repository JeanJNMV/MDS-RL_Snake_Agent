from stable_baselines3 import PPO

MODEL_CLASS = PPO

MODEL_CONFIG = dict(
    policy="MlpPolicy",
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    verbose=0,
)
