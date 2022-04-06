import os
import random

import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.ndimage.filters import gaussian_filter1d

plt.style.use('ggplot')


def epsilon_by_frame(i, epsilon_start, epsilon_final, epsilon_decay):
    return epsilon_final + \
        (epsilon_start - epsilon_final) * \
        np.exp(-1. * i / epsilon_decay)
      
        
def test_fn(env, agent):
    s = env.reset()
    s = np.array(s)    
    ep_reward = 0.
    done = False
    while not done:
        action = agent.get_action(s)
        ns, r, done, _ = env.step(action)
        s = ns
        s = np.array(s)
        ep_reward += r
    return ep_reward


def save(agent, rewards, task, path='./runs/'):
    path = os.path.join(path, task)
    try:
        os.makedirs(path)
    except:
        pass

    agent.q.save(path)

    plt.cla()
    plt.plot(rewards, c='r', alpha=0.3)
    plt.plot(gaussian_filter1d(rewards, sigma=5), c='r', label='Rewards')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative reward')
    plt.title('Branching DDQN: {}'.format(task))
    plt.savefig(os.path.join(path, 'reward.png'))

    pd.DataFrame(rewards, columns=['Reward']).to_csv(
        os.path.join(path, 'rewards.csv'), index=False)

    return path

class DiscreteToContinuous(gym.ActionWrapper):
    def __init__(self, env, action_per_branch):
        super().__init__(env)
        self.action_per_branch = action_per_branch
        low = self.action_space.low
        high = self.action_space.high
        self.mesh = []
        for l, h in zip(low, high):
            self.mesh.append(np.linspace(l, h, action_per_branch))

    def action(self, act):
        # modify act
        act = np.array([self.mesh[i][a] for i, a in enumerate(act)])
        return act
