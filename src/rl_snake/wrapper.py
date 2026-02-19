from gymnasium import Wrapper

from rl_snake.spaces import get_state_encoder
from rl_snake.rewards import get_reward_function


class ModularSnakeWrapper(Wrapper):
    def __init__(self, env, state_type="full_grid", reward_type="dense"):
        super().__init__(env)

        self.state_encoder = get_state_encoder(state_type)
        self.observation_space = self.state_encoder.observation_space

        self.reward_fn = get_reward_function(reward_type)

        self.previous_info = None
        self.steps_in_episode = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        self.previous_info = info
        self.steps_in_episode = 0

        self.reward_fn.reset()

        encoded_obs = self.state_encoder.encode(obs, info)
        return encoded_obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.steps_in_episode += 1

        shaped_reward = self.reward_fn.compute(
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
            previous_info=self.previous_info,
            steps=self.steps_in_episode,
        )

        encoded_obs = self.state_encoder.encode(obs, info)
        self.previous_info = info

        return encoded_obs, shaped_reward, terminated, truncated, info
