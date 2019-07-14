import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register
import random as pr

def rargmax(vector):
    # choose random action when there is no max value
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return pr.choice(indices)

'''
register environment
SFFF
FHFH
FFFH
HFFG
'''
register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={
        'map_name': '4x4',
        'is_slippery': False
    }
)
env = gym.make('FrozenLake-v3')

# make a Q table with numpy (initial values are zeros)
Q = np.zeros([env.observation_space.n, env.action_space.n])
num_episodes = 2000

# update q tables for num_episodes times
r_list = []
for i in range(num_episodes):
    state = env.reset()
    r_all = 0
    done = False

    while not done:
        action = rargmax(Q[state, :])
        new_state, reward, done, _ = env.step(action)
        Q[state, action] = reward + np.max(Q[new_state, :])

        r_all += reward
        state = new_state

    r_list.append(r_all)

# print the result
print("Success rate: " + str(sum(r_list) / num_episodes))
print("Final Q-Table Values")
print("LEFT DOWN RIGHT UP")
print(Q)
plt.bar(range(len(r_list)), r_list, color="blue")
plt.show()
