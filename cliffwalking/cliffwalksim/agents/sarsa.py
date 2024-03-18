from agents.tabularagent import TabularAgent


class SarsaAgent(TabularAgent):
    def update(self, trajectory: tuple) -> None:
        state, action, reward, new_state, time_step = trajectory

        new_action = self.policy(new_state, time_step)
        self.q_table[state][action] += self.learning_rate * (
            reward
            + self.discount_rate * self.q_table[new_state][new_action]
            - self.q_table[state][action]
        )


    def policy(self, state, time_step):
        return self.epsilon_greedy(state, time_step)
