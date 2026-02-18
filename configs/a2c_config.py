from stable_baselines3 import A2C

MODEL_CLASS = A2C

MODEL_CONFIG = dict(
    policy="MlpPolicy",
    learning_rate=7e-4,
    n_steps=5,
    gamma=0.99,
    gae_lambda=1.0,
    ent_coef=0.01,
    vf_coef=0.5,
    verbose=0,
)
