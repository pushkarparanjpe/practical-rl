import gym
import numpy as np
from collections import deque
from matplotlib import pyplot as plt


class BaseExperiment(object):
  '''

    Experiment to train a reinforcement-learning agent that learns using a table of Q-values.
    Agent is a car. Goal is to climb the mountain and reach the flag.
    Following is the set of actions:
      action = 0  # PUSH LEFT
      action = 1  # DO NOTHING
      action = 2  # PUSH RIGHT

    Ref.: https://pythonprogramming.net/q-learning-reinforcement-learning-python-tutorial/

  '''

  # Settings :
  # ----------

  # Settings for discreti-sing the observed continuous state values
  DISCRETE_BUCKET_SIZES = [20, 20]

  # Settings for Q-Learning
  LEARNING_RATE = 0.1
  DISCOUNT = 0.95
  EPISODES = 10_00

  # Settings for rendering, stats
  SHOW_EVERY = 1000
  STATS_EVERY = 100

  # Settings for exploration
  # In initial episodes, epsilon has a high value --> low probability greedy-action-selection
  #   But in later episodes, epsilon will have a low value --> high probability of greedy-action-selection
  epsilon = 1
  START_EPSILON_DECAYING = 1
  END_EPSILON_DECAYING = EPISODES//2
  EPSILON_DECAY_VALUE = epsilon \
                            / (
                                   END_EPSILON_DECAYING
                                 - START_EPSILON_DECAYING
                              )


  def __init__(self):

    # Create the gym environment for the "MountainCar" task
    self.env = gym.make("MountainCar-v0")
    print("HIGH", self.env.observation_space.high)
    print("LOW", self.env.observation_space.low)
    print("n actions", self.env.action_space.n)

    self.DISCRETE_WINDOW_SIZES = (
      self.env.observation_space.high - self.env.observation_space.low
    ) / self.DISCRETE_BUCKET_SIZES
    print("OBS BUCKETS WINDOW SIZES", self.DISCRETE_WINDOW_SIZES)

    # Randomly init the Q-table
    self.q_table = np.random.uniform(
      low=-2, high=0,
      size=(*self.DISCRETE_BUCKET_SIZES, self.env.action_space.n)
    )
    print("Q Table shape: ", self.q_table.shape)

    # Accessories for stats
    self.achieved_rewards = deque(maxlen=self.STATS_EVERY)
    self.stats = {'ep': [], 'min': [], 'avg': [], 'max': []}


  # Helper function that discreti-ses continuous state values to discrete values
  def get_discrete_state(self, state):
    discrete_state = (
      state - self.env.observation_space.low
    ) / self.DISCRETE_WINDOW_SIZES
    return tuple(discrete_state.astype(np.int32))


  def init_episode_state(self):
    # Default Init : places the agent in the env
    #   at a random position and having random velocity
    discrete_state = self.get_discrete_state(self.env.reset())
    return discrete_state


  def loop_over_steps(self, episode):

    discrete_state = self.init_episode_state()

    # Rendering is expensive, don't render every episode,
    #   only render every SHOW_EVERY'th episode
    if episode % self.SHOW_EVERY == 0:
      render = True
      print(episode)
    else:
      render = False

    # Variable to indicate: Has the episode completed ?
    done = False

    # Init the episode reward to 0
    episode_reward = 0

    # Loop over all steps of this episode
    while not done:

      # A. Scheduled epsilon-greedy strategy
      if np.random.random() > self.epsilon:
        # [EXPLOIT]
        # Choose the best action for
        #   this particular discrete state
        action = np.argmax(self.q_table[discrete_state])
      else:
        # [EXPLORE]
        # Choose a random action
        action = np.random.randint(0, self.env.action_space.n)

      # # B. Always greedy strategy
      # action = np.argmax(q_table[discrete_state])


      # Step the env to get:
      #   a new state, a reward, an episode done status, etc.
      new_state, reward, done, _ = self.env.step(action)

      # Accumulate reward for stats
      episode_reward += reward

      # The env sent us to a new state, discretise the new_state
      new_discrete_state = self.get_discrete_state(new_state)

      # Render if it is the SHOW_EVERY'th episode
      if render:
        self.env.render()


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

        # Calculate the new Q value for this particular state and the taken action
        # Ref.: https://pythonprogramming.net/static/images/reinforcement-learning/new-q-value-formula.png
        new_q = (1 - self.LEARNING_RATE) * current_q \
              + self.LEARNING_RATE * (reward + self.DISCOUNT * max_future_q)

      # MountainCar made it to / past the goal flag !
      else:
        # We got the max possible reward (at any step) i.e. 0
        new_q = 0

      # Update the q_table
      self.q_table[discrete_state + (action,)] = new_q

      if done:
        # Collect total total reward for this episode for stats
        self.achieved_rewards.append(episode_reward)

      # Update the current state var to the newly acquired discrete state
      discrete_state = new_discrete_state


  def loop_over_episodes(self):
    # Loop over all episodes
    for episode in range(self.EPISODES):

      self.loop_over_steps(episode)

      # Agg stats
      if episode % self.STATS_EVERY == 0:
        self.stats['ep'].append(episode)
        min_ = np.min(self.achieved_rewards)
        max_ = np.max(self.achieved_rewards)
        avg_ = np.mean(self.achieved_rewards)
        self.stats['min'].append(min_)
        self.stats['max'].append(max_)
        self.stats['avg'].append(avg_)
        print(f"Stats: ep {episode} , min {min_} , max {max_} , avg {avg_}")

      # Decay the epsilon
      if self.START_EPSILON_DECAYING <= episode <= self.END_EPSILON_DECAYING:
        self.epsilon -= self.EPSILON_DECAY_VALUE

    # Close the gym env
    self.env.close()


  def viz_stats(self):
    # Viz stats
    plt.plot(self.stats['ep'], self.stats['min'], label='min rewards')
    plt.plot(self.stats['ep'], self.stats['max'], label='max rewards')
    plt.plot(self.stats['ep'], self.stats['avg'], label='avg rewards')
    plt.legend(loc=0)
    plt.show()


  def run(self):
    self.loop_over_episodes()
    self.viz_stats()


if __name__=='__main__':

  expt = BaseExperiment()
  expt.run()
