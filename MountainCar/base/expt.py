import gym
import numpy as np
from collections import deque
from matplotlib import pyplot as plt


class BaseExperiment(object):
    """

      Experiment to train a reinforcement-learning agent that learns using a table of Q-values.
      Agent is a car. Goal is to climb the mountain and reach the flag.
      Following is the set of actions:
        action = 0  # PUSH LEFT
        action = 1  # DO NOTHING
        action = 2  # PUSH RIGHT

      Ref.: https://pythonprogramming.net/q-learning-reinforcement-learning-python-tutorial/

    """

    # Settings :
    # ----------

    # Settings for discret-ising (binning) the observed continuous state values
    OBS_BIN_COUNTS = [20, 20]  # State is 2-dimensional (position, velocity)

    # Settings for Q-Learning
    LEARNING_RATE = 0.1
    DISCOUNT = 0.95
    EPISODES = 10_000

    # Settings for exploration
    # In initial episodes, epsilon has a high value --> low probability greedy-action-selection
    #   But in later episodes, epsilon will have a low value --> high probability of greedy-action-selection
    epsilon = 1
    START_EPSILON_DECAYING = 1
    END_EPSILON_DECAYING = EPISODES // 2

    # Settings for rendering, stats
    SHOW_EVERY = 1000
    STATS_EVERY = 100

    def __init__(self):

        # Create the gym environment for the "MountainCar" task
        self.env = gym.make("MountainCar-v0")
        print("HIGH", self.env.observation_space.high)
        print("LOW", self.env.observation_space.low)
        print("n actions", self.env.action_space.n)

        self.OBS_BIN_WIDTHS = (self.env.observation_space.high - self.env.observation_space.low) / self.OBS_BIN_COUNTS
        print("OBS BUCKETS BIN WIDTHS", self.OBS_BIN_WIDTHS)

        # Randomly init the Q-table
        self.q_table = np.random.uniform(
            low=-2, high=0,
            size=(*self.OBS_BIN_COUNTS, self.env.action_space.n)
        )
        print("Q Table shape: ", self.q_table.shape)

        # Settings for exploration
        self.EPSILON_DECAY_VALUE = self.epsilon / (self.END_EPSILON_DECAYING - self.START_EPSILON_DECAYING)

        # Accessories for stats
        self.achieved_rewards = deque(maxlen=self.STATS_EVERY)
        self.stats = {'ep': [], 'min': [], 'avg': [], 'max': []}

    # Helper function that discreti-ses continuous state values to discrete values
    def get_discrete_state(self, state):
        discrete_state = (state - self.env.observation_space.low) / self.OBS_BIN_WIDTHS
        return tuple(discrete_state.astype(np.int32))

    def init_episode_state(self):
        # Default Init : places the agent in the env
        #   at a random position and having random velocity
        discrete_state = self.get_discrete_state(self.env.reset())
        return discrete_state

    def choose_action(self, discrete_state):
        # Scheduled epsilon-greedy strategy
        if np.random.random() > self.epsilon:
            # [EXPLOIT]
            # Choose the best action for
            #   this particular discrete state
            action = np.argmax(self.q_table[discrete_state])
        else:
            # [EXPLORE]
            # Choose a random action
            action = np.random.randint(0, self.env.action_space.n)
        return action

    def do_scheduled_epsilon_update(self, episode):
        # Decay the epsilon according to the schedule
        if self.START_EPSILON_DECAYING <= episode <= self.END_EPSILON_DECAYING:
            self.epsilon -= self.EPSILON_DECAY_VALUE

    def do_scheduled_learning_rate_update(self, episode):
        # TODO : impl update to learning_rate
        pass

    def play_episode(self, episode):
        # Acquire the initial state for this episode
        discrete_state = self.init_episode_state()
        # Variable to indicate: Has the episode completed ?
        done = False
        # Init the episode reward to 0
        episode_reward = 0

        # Render SHOW_EVERY'th episode only
        render = True if (episode % self.SHOW_EVERY == 0) else False

        # Loop over all steps of this episode
        while not done:

            # Choose an action
            action = self.choose_action(discrete_state)
            # Step the env
            new_state, reward, done, _ = self.env.step(action)
            # Discret-ise the new state
            new_discrete_state = self.get_discrete_state(new_state)

            # Determine the updated Q-value for (current state, chosen action)
            # ----------------------------------------------------------------
            # Unpack the new_state
            pos, vel = new_state
            # MountainCar did not reach the goal flag
            if pos < self.env.goal_position:
                # Current Q value for this particular state and the taken action
                current_q = self.q_table[discrete_state + (action,)]
                # Max possible Q value by actioning from the future state
                max_future_q = np.max(self.q_table[new_discrete_state])
                # Calculate the new Q value for this particular (state, taken action)
                # Ref.: https://pythonprogramming.net/static/images/reinforcement-learning/new-q-value-formula.png
                new_q = (1 - self.LEARNING_RATE) * current_q \
                        + self.LEARNING_RATE * (reward + self.DISCOUNT * max_future_q)
            # MountainCar made it to / past the goal flag !
            else:
                # We got the max possible reward (at any step) i.e. 0
                new_q = 0
            # ----------------------------------------------------------------

            # Update the q_table
            self.q_table[discrete_state + (action,)] = new_q

            # Accumulate reward for stats
            episode_reward += reward

            # Did this episode complete ?
            if done:
                # Collect total total reward for this episode for stats
                self.achieved_rewards.append(episode_reward)

            # Set new discrete state as the current discrete state
            discrete_state = new_discrete_state

            # Render if it is the SHOW_EVERY'th episode
            if render:
                self.env.render()

    def agg_stats(self, episode):
        if episode % self.STATS_EVERY == 0:
            self.stats['ep'].append(episode)
            min_ = np.min(self.achieved_rewards)
            max_ = np.max(self.achieved_rewards)
            avg_ = np.mean(self.achieved_rewards)
            self.stats['min'].append(min_)
            self.stats['max'].append(max_)
            self.stats['avg'].append(avg_)
            print(f"Stats: ep {episode} , min {min_} , max {max_} , avg {avg_}")

    def train(self):
        # Loop over all episodes
        for episode in range(self.EPISODES):
            # Update epsilon
            self.do_scheduled_epsilon_update(episode)
            # Update learning rate
            self.do_scheduled_learning_rate_update(episode)
            # Run the intra-episode loop
            self.play_episode(episode)
            # Agg stats
            self.agg_stats(episode)
        # Close the gym env
        self.env.close()

    def viz_stats(self):
        # Viz stats
        plt.plot(self.stats['ep'], self.stats['min'], label='min rewards')
        plt.plot(self.stats['ep'], self.stats['max'], label='max rewards')
        plt.plot(self.stats['ep'], self.stats['avg'], label='avg rewards')
        plt.ylabel("agg rewards")
        plt.xlabel("episodes")
        plt.title(self.__class__.__name__)
        plt.legend(loc=0)
        plt.savefig(f"charts/stats_{self.__class__.__name__}", dpi=90)
        plt.show()

    def run(self):
        self.train()
        self.viz_stats()


if __name__ == '__main__':
    expt = BaseExperiment()
    expt.run()
