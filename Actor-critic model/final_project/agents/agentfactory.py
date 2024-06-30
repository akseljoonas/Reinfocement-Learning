import gymnasium as gym

from final_project.agents.abstractagent import AbstractAgent
from final_project.agents.actor_critic_agent import ActorCriticAgent
from final_project.agents.randomagent import RandomAgent


class AgentFactory:
    """
    Naive factory method implementation for
    RL agent creation.
    """

    @staticmethod
    def create_agent(agent_type: str, env: gym.Env) -> AbstractAgent:
        """
        Factory method for Agent creation.
        :param env: gymnasium environment.
        :param agent_type: a string key corresponding to the agent.
        :return: an object of type Agent.
        """
        obs_space = env.observation_space
        action_space = env.action_space

        if agent_type == "ACTOR-CRITIC-AGENT":
            return ActorCriticAgent(obs_space, action_space)    # CHANGE THIS
        elif agent_type == "RANDOM":
            return RandomAgent(obs_space, action_space)

        raise ValueError("Invalid agent type")
