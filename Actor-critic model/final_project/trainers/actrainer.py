import torch
from torch import nn
from torchrl.data import ReplayBuffer
from final_project.models.twoheadedmlp import TwoHeadedMLP
from final_project.trainers.abstracttrainer import Trainer
from final_project.util.device import fetch_device
from final_project.models.sampling import log_prob_policy

""""
IMPORTANT: in pseudocode gradient ascent is performed. But PyTorch automatic differentiation
facilities perform gradient descent by default. Therefore, you should reverse the signs to turn gradient ascent
in the pseudocode to gradient descent.
"""


class ACTrainer(Trainer):
    """
    One-step Actor-Critic Trainer based on Sutton and Barto's algorithm.
    """

    def __init__(
        self,
        buf: ReplayBuffer,
        actor_model: TwoHeadedMLP,
        critic_model: nn.Module,
        learning_rate_actor: float,
        learning_rate_critic: float,
        discount_factor: float,
    ):
        """
        Initialize the Actor-Critic Trainer.

        :param buf: ReplayBuffer for storing experiences.
        :param actor_model: The actor model (policy).
        :param critic_model: The critic model (value function).
        :param learning_rate_actor: Learning rate for the actor.
        :param learning_rate_critic: Learning rate for the critic.
        :param discount_factor: Discount factor for future rewards.
        """
        self.device = fetch_device()
        self.actor_model = actor_model
        self.critic_model = critic_model
        self.discount_factor = discount_factor
        self.I = 1

        self.buf = buf

        # Optimizes policy parameters
        self.actor_optimizer = torch.optim.Adam(
            self.actor_model.parameters(), lr=learning_rate_actor
        )
        # Optimizes critic parameters
        self.critic_optimizer = torch.optim.Adam(
            self.critic_model.parameters(), lr=learning_rate_critic
        )

    def _trajectory(self) -> tuple:
        """
        Sample the latest trajectory from the replay buffer.

        :return: A tuple containing the states, actions, rewards, and next states.
        """
        trajectories = self.buf.sample(batch_size=1)
        return trajectories[0], trajectories[1], trajectories[2], trajectories[3]

    def train(self) -> None:
        # Sample a trajectories
        states, actions, rewards, next_states = self._trajectory()

        # Forward pass through the critic to get value estimates
        current_values = self.critic_model(states)
        next_values = self.critic_model(next_states)

        # Compute the TD error
        deltas = rewards + self.discount_factor * next_values - current_values

        # Compute the critic loss
        critic_loss = torch.mean(deltas)

        # Backward pass to get the gradient
        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        # Perform a single optimization step
        self.critic_optimizer.step()

        # Actor update
        log_probs = log_prob_policy(self.actor_model, states, actions)

        # Compute the policy gradient
        actor_loss = -torch.mean(
            log_probs * deltas.detach()
        )  # Negative for gradient ascent

        # Backward pass to compute the gradient
        self.actor_optimizer.zero_grad()
        actor_loss.backward()

        # Update the actor's parameters
        self.actor_optimizer.step()

        # We do not use this, but it was in the pseudocode so I still update it
        self.I *= self.discount_factor

        return actor_loss, critic_loss  # for plotting
