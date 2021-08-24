from MountainCar.base.expt import BaseExperiment
import numpy as np


class AlwaysGreedyStrategyExpt(BaseExperiment):
    # Settings for Q-Learning
    EPISODES = 10_000

    def choose_action(self, discrete_state):
        # Always greedy strategy
        action = np.argmax(self.q_table[discrete_state])
        return action


if __name__ == '__main__':
    expt = AlwaysGreedyStrategyExpt()
    expt.run()
