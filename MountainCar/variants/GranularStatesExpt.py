from MountainCar.base.expt import BaseExperiment


class GranularStatesExpt(BaseExperiment):
    # Settings for discreti-sing the observed continuous state values
    OBS_BIN_COUNTS = [40, 40]

    # Settings for Q-Learning
    EPISODES = 10_000


if __name__ == '__main__':
    expt = GranularStatesExpt()
    expt.run()
