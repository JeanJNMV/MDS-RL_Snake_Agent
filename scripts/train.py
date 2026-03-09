import logging

import gymnasium as gym
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from rl_snake.agents import BaseAgent
from rl_snake.rewards import BaseReward
from rl_snake.spaces import BaseStateEncoder

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="train")
def main(cfg: DictConfig) -> None:
    log_level = cfg.get("log_level", "INFO")
    logging.basicConfig(
        level=getattr(logging, str(log_level).upper()),
        format="%(message)s",
        force=True,
    )

    env = gym.make("Snake-v1")

    agent: BaseAgent = instantiate(cfg.agent)
    encoder: BaseStateEncoder = instantiate(cfg.encoder)
    reward_wrapper: BaseReward = instantiate(cfg.reward)

    shaped_episode_rewards = []
    raw_episode_rewards = []
    episode_lengths = []

    for episode in range(cfg.episodes):
        if cfg.seed is None:
            obs, info = env.reset()
        else:
            obs, info = env.reset(seed=cfg.seed + episode)

        reward_wrapper.reset()

        state = encoder.encode(obs, info)

        done = False
        shaped_episode_reward = 0.0
        raw_episode_reward = 0.0
        steps = 0

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
            shaped_episode_reward += float(shaped_reward)
            raw_episode_reward += float(raw_reward)
            steps += 1

        shaped_episode_rewards.append(shaped_episode_reward)
        raw_episode_rewards.append(raw_episode_reward)
        episode_lengths.append(steps)

        if (episode + 1) % cfg.log_every == 0:
            mean_shaped_reward = (
                sum(shaped_episode_rewards[-cfg.log_every :]) / cfg.log_every
            )
            mean_raw_reward = sum(raw_episode_rewards[-cfg.log_every :]) / cfg.log_every
            mean_steps = sum(episode_lengths[-cfg.log_every :]) / cfg.log_every
            logger.info(
                "Episode %4d/%d | Avg Raw Reward (%d ep): %8.3f | Avg Shaped Reward (%d ep): %8.3f | Avg Steps (%d ep): %6.1f",
                episode + 1,
                cfg.episodes,
                cfg.log_every,
                mean_raw_reward,
                cfg.log_every,
                mean_shaped_reward,
                cfg.log_every,
                mean_steps,
            )

    if shaped_episode_rewards:
        overall_shaped_reward = sum(shaped_episode_rewards) / len(
            shaped_episode_rewards
        )
        overall_raw_reward = sum(raw_episode_rewards) / len(raw_episode_rewards)
        overall_steps = sum(episode_lengths) / len(episode_lengths)
        logger.info(
            "Training complete | Episodes: %d | Mean Raw Reward: %8.3f | Mean Shaped Reward: %8.3f | Mean Steps: %6.1f",
            len(shaped_episode_rewards),
            overall_raw_reward,
            overall_shaped_reward,
            overall_steps,
        )

    if cfg.save.enabled:
        agent.save(cfg.save.path)
        logger.info("Agent saved to %s", cfg.save.path)

    env.close()


if __name__ == "__main__":
    main()
