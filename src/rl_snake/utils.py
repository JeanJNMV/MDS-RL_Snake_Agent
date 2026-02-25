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
    """Training progress monitor with correct food detection."""

    def __init__(self, log_interval=100):
        super().__init__()
        self.log_interval = log_interval

        self.episode_rewards = []
        self.episode_lengths = []

        # Track food per env (supports vec env)
        self.prev_food_positions = {}
        self.total_food = 0

    def _on_step(self):
        infos = self.locals.get("infos", [])
        # rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])

        for i, info in enumerate(infos):
            if "head" in info and "food" in info:
                head_pos = info["head"]
                food_pos = info["food"]

                # Detect food eaten:
                # If previous food position exists and head == previous food
                if i in self.prev_food_positions:
                    if head_pos == self.prev_food_positions[i]:
                        self.total_food += 1

                # Update stored food position
                self.prev_food_positions[i] = food_pos

        # Handle episode end
        for i, done in enumerate(dones):
            if done:
                info = infos[i]

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
                            f"Total Food: {self.total_food:6d}"
                        )

        return True
