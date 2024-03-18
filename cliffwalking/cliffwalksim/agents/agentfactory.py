import gymnasium as gym

from agents.randomagent import RandomAgent
from agents.sarsa import SarsaAgent
from agents.q_learning import QAgent
from agents.double_q import DoubleQAgent

from agents.tabularagent import TabularAgent


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
        lr = 0.25
        dr = 0.7
        if agent_type == "SARSA":
            return SarsaAgent(obs_space, action_space, lr, dr)
        elif agent_type == "Q-LEARNING":
            return QAgent(obs_space, action_space, lr, dr)
        elif agent_type == "DOUBLE-Q-LEARNING":
            return DoubleQAgent(obs_space, action_space, lr, dr)
        elif agent_type == "RANDOM":
            return RandomAgent(obs_space, action_space, lr, dr)

        raise ValueError("Invalid agent type")
