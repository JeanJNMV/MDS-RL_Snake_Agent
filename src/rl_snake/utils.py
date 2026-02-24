import importlib
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


def load_config(model_name: str):
    """Dynamically loads the model class and configuration based on the provided model name."""
    module = importlib.import_module(f"configs.{model_name}_config")
    return module.MODEL_CLASS, module.MODEL_CONFIG


def load_model_class(model_name: str):
    """Dynamically loads the model class based on the provided model name."""
    module = importlib.import_module(f"configs.{model_name}_config")
    return module.MODEL_CLASS


class TrainingMonitor(BaseCallback):
    """Simple training progress monitor"""

    def __init__(self, log_interval=100):
        super().__init__()
        self.log_interval = log_interval
        self.episode_rewards = []
        self.episode_lengths = []
        self.food_count = 0

    def _on_step(self):
        reward = self.locals.get("rewards", [0])[0]

        if reward > 100:
            self.food_count += 1

        if self.locals.get("dones", [False])[0]:
            info = self.locals["infos"][0]
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])

                if len(self.episode_rewards) % self.log_interval == 0:
                    recent_rewards = self.episode_rewards[-self.log_interval :]
                    recent_lengths = self.episode_lengths[-self.log_interval :]

                    print(
                        f"Episodes: {len(self.episode_rewards):5d} | "
                        f"Avg Reward: {np.mean(recent_rewards):7.2f} | "
                        f"Avg Length: {np.mean(recent_lengths):6.1f} | "
                        f"Total Food: {self.food_count:5d}",
                    )

        return True
