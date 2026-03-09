import gymnasium as gym
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from rl_snake.agents import BaseAgent
from rl_snake.rewards import BaseReward
from rl_snake.spaces import BaseStateEncoder


@hydra.main(version_base=None, config_path="../conf", config_name="test")
def main(cfg: DictConfig) -> None:
    env = gym.make("Snake-v1", render_mode="human", fps=cfg.fps)

    agent: BaseAgent = instantiate(cfg.agent)
    encoder: BaseStateEncoder = instantiate(cfg.encoder)
    reward_wrapper: BaseReward = instantiate(cfg.reward)

    obs, info = env.reset()
    state = encoder.encode(obs, info)

    done = False
    while not done:
        action = agent.choose_action(state)

        next_obs, raw_reward, terminated, truncated, next_info = env.step(action)

        next_state = encoder.encode(next_obs, next_info)
        shaped_reward = reward_wrapper.compute(
            state=state,
            action=action,
            next_state=next_state,
            raw_reward=raw_reward,
            terminated=terminated,
            truncated=truncated,
            info=next_info,
        )

        done = terminated or truncated

        agent.update(
            state=state,
            action=action,
            reward=shaped_reward,
            next_state=next_state,
            done=done,
        )

        state = next_state


if __name__ == "__main__":
    main()
