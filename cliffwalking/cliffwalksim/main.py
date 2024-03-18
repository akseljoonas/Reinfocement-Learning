import gymnasium as gym
import numpy as np
from tqdm import tqdm

from agents.agentfactory import AgentFactory
from util.metricstracker import MetricsTracker


def env_interaction(
    env_str: str, agent_type: str, tracker: MetricsTracker, num_episodes: int = 500 
) -> None:
    env = gym.make(env_str)
    obs, info = env.reset()
    agent = AgentFactory.create_agent(agent_type, env=env)

    returns = []  
    rewards_sum = 0
    best_rewards_sum = 0
    time_step = 0
    while True:
        old_obs = obs
        time_step += 1
        action = agent.policy(obs, time_step)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards_sum += reward
        agent.update((old_obs, action, reward, obs, time_step))

        if terminated or truncated:
            returns.append(rewards_sum)
            rewards_sum = 0
            num_episodes -= 1
            # Episode ended.
            obs, info = env.reset()

        if num_episodes == 0:
            tracker.record_return(agent_type, np.mean(returns))
            break

    
    env.close()


if __name__ == "__main__":
    agents = ("SARSA", "Q-LEARNING", "DOUBLE-Q-LEARNING")
    tracker = MetricsTracker()

    for _ in tqdm(range(500)):
        for agent in agents:
            rewards_sum = env_interaction("CliffWalking-v0", agent, tracker, 500)
            

    tracker.plot()