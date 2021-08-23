from MountainCar.base.expt import BaseExperiment
import numpy as np


class FixedInitStateExpt(BaseExperiment):
    # Settings for Q-Learning
    EPISODES = 10_000

    def init_episode_state(self):
        # A. Default Init : places the agent in the env
        #      at a random position and having random velocity
        discrete_state = self.get_discrete_state(self.env.reset())

        # A+B. Override the default init to always start the agent in a fixed state
        discrete_state = self.get_discrete_state(np.array([-0.4, 0]))
        self.env.env.state = np.array([-0.4, 0])

        return discrete_state


if __name__ == '__main__':
    expt = FixedInitStateExpt()
    expt.run()
