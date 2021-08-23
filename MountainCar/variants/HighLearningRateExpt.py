from MountainCar.base.expt import BaseExperiment


class HighLearningRateExpt(BaseExperiment):
    # Settings for Q-Learning
    LEARNING_RATE = 0.8
    EPISODES = 10_000


if __name__ == '__main__':
    expt = HighLearningRateExpt()
    expt.run()
