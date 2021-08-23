from MountainCar.base.expt import BaseExperiment


class LowDiscountingExpt(BaseExperiment):
    # Settings for Q-Learning
    DISCOUNT = 0.1
    EPISODES = 10_000


if __name__ == '__main__':
    expt = LowDiscountingExpt()
    expt.run()
