import gymnasium as gym

from cliffwalksim.agents.randomagent import RandomAgent
from cliffwalksim.agents.tabularagent import TabularAgent


class AgentFactory:
    """
    Naive factory method implementation for
    RL agent creation.
    """

    @staticmethod
    def create_agent(agent_type: str, env: gym.Env) -> TabularAgent:
        """
        Factory method for Agent creation.
        :param env: gymnasium environment.
        :param agent_type: a string key corresponding to the agent.
        :return: an object of type Agent.
        """
        obs_space = env.observation_space
        action_space = env.action_space

        if agent_type == "SARSA":
            return RandomAgent(obs_space, action_space)    # CHANGE THIS
        elif agent_type == "Q-LEARNING":
            return RandomAgent(obs_space, action_space)
        elif agent_type == "DOUBLE-Q-LEARNING":
            return RandomAgent(obs_space, action_space)
        elif agent_type == "RANDOM":
            return RandomAgent(obs_space, action_space)

        raise ValueError("Invalid agent type")
