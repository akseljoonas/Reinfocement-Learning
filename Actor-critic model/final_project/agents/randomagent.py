from final_project.agents.abstractagent import AbstractAgent


class RandomAgent(AbstractAgent):
    def add_trajectory(self, trajectory):
        pass

    def update(self) -> None:
        pass

    def policy(self, state):
        return self.action_space.sample()

    def save(self, file_path='./') -> None:
        pass

    def load(self, file_path='./') -> None:
        pass
