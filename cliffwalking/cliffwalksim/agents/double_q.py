from gymnasium.spaces.discrete import Discrete
import numpy as np
from agents.tabularagent import TabularAgent


class DoubleQAgent(TabularAgent):
    def __init__(
        self,
        state_space: Discrete,
        action_space: Discrete,
        learning_rate=0.1,
        discount_rate=0.9,
    ):
        super().__init__(state_space, action_space, learning_rate, discount_rate)
        self.q_1 = self.q_table
        self.q_2 = self.q_table

    def update(self, trajectory: tuple) -> None:
        state, action, reward, new_state, _ = trajectory

        if np.random.rand() > 0.5:
            self.q_1[state][action] += self.learning_rate * (
                reward
                + self.discount_rate
                * self.q_2[new_state][np.argmax(self.q_1[new_state])]
                - self.q_1[state][action]
            )
        else:
            self.q_2[state][action] += self.learning_rate * (
                reward
                + self.discount_rate
                * self.q_1[new_state][np.argmax(self.q_2[new_state])]
                - self.q_2[state][action]
            )




    def policy(self, state, time_step):
        return self.epsilon_greedy_double(state, time_step)

    def epsilon_greedy_double(self, state, time_step) -> int:
        epsilon = (
            1 / (time_step + 0.000001)  # Addition to avoid ZeroDivisionError
        )
        if np.random.random() < epsilon:
            return self.env_action_space.sample()
        else:
            return np.argmax(self.q_1[state] + self.q_2[state])
