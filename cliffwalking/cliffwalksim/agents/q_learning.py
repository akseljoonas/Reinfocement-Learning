from agents.tabularagent import TabularAgent


class QAgent(TabularAgent):
    def update(self, trajectory: tuple) -> None:
        state, action, reward, new_state, _ = trajectory

        self.q_table[state][action] += self.learning_rate * (
            reward
            + self.discount_rate * max(self.q_table[new_state])
            - self.q_table[state][action]
        )
        

    def policy(self, state, time_step):
        return self.epsilon_greedy(state, time_step)
