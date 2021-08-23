from MountainCar.base.expt import BaseExperiment


class MedDiscountingExpt(BaseExperiment):
    # Settings for Q-Learning
    DISCOUNT = 0.5
    EPISODES = 10_000


if __name__ == '__main__':
    expt = MedDiscountingExpt()
    expt.run()
