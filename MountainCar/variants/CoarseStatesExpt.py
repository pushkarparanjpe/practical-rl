from MountainCar.base.expt import BaseExperiment


class CoarseStatesExpt(BaseExperiment):
    # Settings for discreti-sing the observed continuous state values
    OBS_BIN_COUNTS = [10, 10]

    # Settings for Q-Learning
    EPISODES = 10_000


if __name__ == '__main__':
    expt = CoarseStatesExpt()
    expt.run()
