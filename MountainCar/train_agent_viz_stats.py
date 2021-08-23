'''
  Ref.: https://pythonprogramming.net/q-learning-reinforcement-learning-python-tutorial/


  Notes:
    action = 0  # PUSH LEFT
    action = 1  # DO NOTHING
    action = 2  # PUSH RIGHT

'''

import gym
import numpy as np
from collections import deque
from matplotlib import pyplot as plt



# Create the gym environment for the "MountainCar" task
env = gym.make("MountainCar-v0")
print("HIGH", env.observation_space.high)
print("LOW", env.observation_space.low)
print("n actions", type(env.action_space.n))


# Settings for discreti-sing the observed continuous state values
DISCRETE_BUCKET_SIZES = [20, 20]
DISCRETE_WINDOW_SIZES = (
  env.observation_space.high - env.observation_space.low
) / DISCRETE_BUCKET_SIZES
print("OBS BUCKETS WINDOW SIZES", DISCRETE_WINDOW_SIZES)


# Q-Learning settings
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 10_000


# Exploration settings
# In initial episodes, epsilon has a high value --> low probability greedy-action-selection
#   But in later episodes, epsilon will have a low value --> high probability of greedy-action-selection
epsilon = 1
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES//2
episilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)


# Randomly init the Q-table
q_table = np.random.uniform(
  low=-2, high=0,
  size=(*DISCRETE_BUCKET_SIZES, env.action_space.n)
)
print("Q Table shape: ", q_table.shape)


# Helper function that discreti-ses continuous state values to discrete values
def get_discrete_state(state):
  discrete_state = (
    state - env.observation_space.low
  ) / DISCRETE_WINDOW_SIZES
  return tuple(discrete_state.astype(np.int32))


# Settings, accessories for rendering, stats
SHOW_EVERY = 1000
STATS_EVERY = 100
achieved_rewards = deque(maxlen=STATS_EVERY)
stats = {'ep': [], 'min': [], 'avg': [], 'max': []}


# Loop over all episodes
for episode in range(EPISODES):

  # Rendering is expensive, don't render every episode,
  #   only render every SHOW_EVERY'th episode
  if episode % SHOW_EVERY == 0:
    render = True
    print(episode)
  else:
    render = False

  # A. Default Init : places the agent in the env
  #      at a random position and having random velocity
  discrete_state = get_discrete_state(env.reset())

  # # B. Override the default init to always start the agent in a fixed state
  # discrete_state = get_discrete_state(np.array([-0.4, 0]))
  # env.env.state = np.array([-0.4, 0])

  # Variable to indicate: Has the episode completed ?
  done = False

  # Init the episode reward to 0
  episode_reward = 0

  # Loop over all steps of this episode
  while not done:

    # A. Scheduled epsilon-greedy strategy
    if np.random.random() > epsilon:
      # [EXPLOIT]
      # Choose the best action for
      #   this particular discrete state
      action = np.argmax(q_table[discrete_state])
    else:
      # [EXPLORE]
      # Choose a random action
      action = np.random.randint(0, env.action_space.n)

    # # B. Always greedy strategy
    # action = np.argmax(q_table[discrete_state])


    # Step the env to get:
    #   a new state, a reward, an episode done status, etc.
    new_state, reward, done, _ = env.step(action)

    # Accumulate reward for stats
    episode_reward += reward

    # The env sent us to a new state, discretise the new_state
    new_discrete_state = get_discrete_state(new_state)

    # Render if it is the SHOW_EVERY'th episode
    if render:
      env.render()

    # Unpack the new_state
    pos, vel = new_state

    # MountainCar did not reach the goal flag
    if pos < env.goal_position:
      # Bunch of calculations leading up to Q-table update ...
      # ------------------------------------------------------

      # Current Q value for this particular state and the taken action
      current_q = q_table[discrete_state + (action,)]

      # Max possible Q value by actioning from the future state
      max_future_q = np.max(q_table[new_discrete_state])

      # Calculate the new Q value for this particular state and the taken action
      # Ref.: https://pythonprogramming.net/static/images/reinforcement-learning/new-q-value-formula.png
      new_q = (1 - LEARNING_RATE) * current_q \
            + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

      # Update the q_table
      q_table[discrete_state + (action,)] = new_q

    # MountainCar made it to / past the goal flag !
    else:
      # We got the max possible reward (at any step) i.e. 0
      #   update the q_table
      q_table[discrete_state + (action,)] = 0


    if done:
      # Collect total total reward for this episode for stats
      achieved_rewards.append(episode_reward)


    # Update the current state var to the newly acquired discrete state
    discrete_state = new_discrete_state


  # Agg stats
  if episode % STATS_EVERY == 0:
    stats['ep'].append(episode)
    min_ = np.min(achieved_rewards)
    max_ = np.max(achieved_rewards)
    avg_ = np.mean(achieved_rewards)
    stats['min'].append(min_)
    stats['max'].append(max_)
    stats['avg'].append(avg_)
    print(f"Stats: ep {episode} , min {min_} , max {max_} , avg {avg_}")

  # Decay the epsilon
  if START_EPSILON_DECAYING <= episode <= END_EPSILON_DECAYING:
    epsilon -= episilon_decay_value


# Close the gym env
env.close()


# Viz stats
plt.plot(stats['ep'], stats['min'], label='min rewards')
plt.plot(stats['ep'], stats['max'], label='max rewards')
plt.plot(stats['ep'], stats['avg'], label='avg rewards')
plt.legend(loc=0)
plt.show()
