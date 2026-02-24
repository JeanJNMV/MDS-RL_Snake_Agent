from gymnasium import Wrapper

from rl_snake.spaces import get_state_encoder
from rl_snake.rewards import get_reward_function


class ModularSnakeWrapper(Wrapper):
    """
    A modular wrapper for Snake RL environments that encodes observations and shapes rewards.

    This wrapper provides flexible observation encoding and reward shaping for a Snake game
    environment. It decouples the state representation and reward computation logic from the
    base environment, allowing for easy experimentation with different state encodings and
    reward functions.

    Attributes:
        state_encoder: An encoder instance that transforms raw observations into the desired
                       state representation format.
        observation_space: The observation space of the encoded state, as defined by the
                          state encoder.
        reward_fn: A reward function instance that computes shaped rewards based on environment
                  feedback and gameplay information.
        previous_info: Dictionary containing information from the previous step, used for
                       computing reward shaping.
        steps_in_episode: Counter tracking the number of steps taken in the current episode.

    Args:
        env: The base Gym environment to wrap.
        state_type (str, optional): The type of state encoding to use. Defaults to "full_grid".
        reward_type (str, optional): The type of reward shaping function to use. Defaults to "dense".

    Methods:
        reset: Resets the environment and returns the encoded initial observation.
        step: Executes an action, applies state encoding and reward shaping, and returns the
              encoded observation and shaped reward.
    """

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
