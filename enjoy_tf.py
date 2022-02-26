#!/usr/bin/env python3
from tqdm import tqdm
import tensorflow as tf
from utils import AgentConfig, BranchingTensorEnv_tf

from bdq_tf import BranchingDQN

args = 'BipedalWalker-v3'

bins = 6
env = BranchingTensorEnv_tf('BipedalWalker-v3', bins)

agent = BranchingDQN([128, 128], [128], [128], env.observation_space.shape[0], env.action_space.shape[0], bins, config=AgentConfig())

agent.q = tf.keras.models.load_model('/home/anas/Documents/learning/BranchingDQN/runs/BipedalWalker-v3_tf')

print(agent)
for ep in tqdm(range(10)):

    s = env.reset()
    done = False
    ep_reward = 0
    while not done:

        action = agent.get_action(s)
        print(action)
        s, r, done, _ = env.step(action)

        env.render()
        ep_reward += r

    print('Ep reward: {:.3f}'.format(ep_reward))

env.close()
