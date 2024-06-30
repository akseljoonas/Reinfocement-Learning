import gymnasium as gym
import torch
import numpy as np

from final_project.agents.abstractagent import AbstractAgent
from final_project.models.mlp import MLP
from final_project.models.mlpmultivariategaussian import MLPMultivariateGaussian
from final_project.models.sampling import sample_two_headed_gaussian_model
from final_project.trainers.actrainer import ACTrainer
from final_project.util.device import fetch_device


class ActorCriticAgent(AbstractAgent):
    def __init__(
        self,
        state_space: gym.spaces.Box,
        action_space: gym.spaces.Box,
        discount_factor: float = 0.9,
        learning_rate_actor: float = 0.001,
        learning_rate_critic: float = 0.01,
    ):
        super().__init__(state_space, action_space, discount_factor)
        """
        Initialize the Actor-Critic Agent.
        
        NOTE: One rule of thumb for the learning rates is that the learning rate of the actor should be lower
        than the critic. Intuitively because the estimated values of the critic are based on past policies,
        so the actor cannot "get ahead" of the critic.

        :param state_space: The state space of the environment.
        :param action_space: The action space of the environment.
        :param discount_factor: Discount factor for future rewards.
        :param learning_rate_actor: Learning rate for the actor model.
        :param learning_rate_critic: Learning rate for the critic model.
        """
        # Once these models are trained, you might want to set them to evaluation mode during evaluation
        self._actor_model = MLPMultivariateGaussian(
            input_size=state_space.shape[0], output_size=action_space.shape[0]
        ).to(device=fetch_device())
        # output_size=1 because value function returns a scalar value.
        self._critic_model = MLP(input_size=state_space.shape[0], output_size=1).to(
            device=fetch_device()
        )

        self._trainer = ACTrainer(
            self._replay_buffer,
            self._actor_model,
            self._critic_model,
            learning_rate_actor,
            learning_rate_critic,
            discount_factor,
        )

        self.device = fetch_device()

    def add_trajectory(self, trajectory: tuple) -> None:
        """
        Add a trajectory to the replay buffer.

        NOTE: One-step Actor Critic does not by default use a replay buffer.
        Therefore, the replay buffer is assumed to have a size of 1 which means
        it will only store the latest trajectory.
        The trainer will later sample from the buffer to retrieve the trajectory
        to apply the update rule.

        :param trajectory: The trajectory to add to the replay buffer.
        """
        state, action, reward, next_state = trajectory
        state_t = torch.as_tensor(state, device=self.device, dtype=torch.float64)
        action_t = torch.as_tensor(action, device=self.device, dtype=torch.float64)
        reward_t = torch.as_tensor(reward, device=self.device, dtype=torch.float64)
        next_state_t = torch.as_tensor(
            next_state, device=self.device, dtype=torch.float64
        )
        self._replay_buffer.add((state_t, action_t, reward_t, next_state_t))

    def update(self) -> None:
        """
        Perform a gradient descent step on both actor (policy) and critic (value function).
        """
        return self._trainer.train()

    def policy(self, state) -> np.array:
        """
        Get the action to take based on the current state.

        :param state: The current state of the environment.
        :return: The action to take.
        """
        state = torch.from_numpy(state).to(
            device=self.device, dtype=torch.float64
        )  # just to make it a Tensor obj

        action, _ = sample_two_headed_gaussian_model(self._actor_model, state)

        return (
            action.cpu().numpy()
        )  # Put the tensor back on the CPU (if applicable) and convert to numpy array.

    def save(self, file_path="./") -> None:
        """
        Save the actor and critic models.

        :param file_path: The directory path to save the models.
        """
        torch.save(self._actor_model.state_dict(), file_path + "actor_model.pth")
        torch.save(self._critic_model.state_dict(), file_path + "critic_model.pth")

    def load(self, file_path="./") -> None:
        """
        Load the actor and critic models.

        :param file_path: The directory path to load the models from.
        """
        self._actor_model.load_state_dict(torch.load(file_path + "actor_model.pth"))
        self._critic_model.load_state_dict(torch.load(file_path + "critic_model.pth"))
