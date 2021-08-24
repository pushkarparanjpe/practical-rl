from MountainCar.base.expt import BaseExperiment


class CoarseVelocityStateExpt(BaseExperiment):
    # Settings for discreti-sing the observed continuous state values
    OBS_BIN_COUNTS = [20, 4]  # State is 2-dimensional (position, velocity)

    # Settings for Q-Learning
    EPISODES = 10_000


if __name__ == '__main__':
    expt = CoarseVelocityStateExpt()
    expt.run()
