from MountainCar.base.expt import BaseExperiment


class LowLearningRateExpt(BaseExperiment):
    # Settings for Q-Learning
    LEARNING_RATE = 0.01
    EPISODES = 10_000


if __name__ == '__main__':
    expt = LowLearningRateExpt()
    expt.run()
