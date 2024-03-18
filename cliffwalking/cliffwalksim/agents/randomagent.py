from agents.tabularagent import TabularAgent


class RandomAgent(TabularAgent):
    def update(self, trajectory: tuple) -> None:
        pass

    def policy(self, state, _):
        return self.env_action_space.sample()
