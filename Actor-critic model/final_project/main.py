import gymnasium as gym
from final_project.agents.agentfactory import AgentFactory
from final_project.util.metricstracker import MetricsTracker


def env_interaction(env_str: str, agent_type: str, num_episodes: int) -> None:
    """
    Simulates interaction between an agent and an environment for a given number of episodes.

    :param env_str: The environment string specifying the environment to use.
    :param agent_type: The type of agent to use for interaction.
    :param num_episodes: The number of episodes to simulate interaction for.
    """
    env = gym.make(env_str)
    obs, info = env.reset()
    agent = AgentFactory.create_agent(agent_type, env=env)
    tracker = MetricsTracker()

    episode_return: float = 0
    while num_episodes > 0:
        old_obs = obs

        action = agent.policy(obs)

        obs, reward, terminated, truncated, info = env.step(action)
        episode_return += reward

        agent.add_trajectory((old_obs, action, reward, obs))

        if agent_type != "RANDOM":  # To avoid errors with Random agent
            agent_loss, critic_loss = agent.update()
        else:
            agent.update()

        if terminated or truncated:
            num_episodes -= 1
            tracker.record_return(agent_id=agent_type, return_val=episode_return)

            if agent_type != "RANDOM":
                tracker.record_loss(
                    agent_id="Actor loss", loss=agent_loss.detach().numpy()
                )
                tracker.record_loss(
                    agent_id="Critic loss", loss=critic_loss.detach().numpy()
                )

            episode_return = 0

            obs, info = env.reset()

    tracker.plot()
    env.close()


if __name__ == "__main__":
    env_interaction("InvertedPendulum-v4", "RANDOM", 1000)
    env_interaction("InvertedPendulum-v4", "ACTOR-CRITIC-AGENT", 1000)
