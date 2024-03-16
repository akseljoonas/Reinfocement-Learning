import gymnasium as gym

from agents.agentfactory import AgentFactory


def env_interaction(env_str: str, agent_type: str, time_steps: int = 1000) -> None:
    env = gym.make(env_str, render_mode='human')
    obs, info = env.reset()
    agent = AgentFactory.create_agent(agent_type, env=env)

    for _ in range(time_steps):
        old_obs = obs
        action = agent.policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        agent.update((old_obs, action, reward, obs))

        if terminated or truncated:
            # Episode ended.
            obs, info = env.reset()
            break

    env.close()


if __name__ == "__main__":
    env_interaction("CliffWalking-v0", 'RANDOM', 20)
