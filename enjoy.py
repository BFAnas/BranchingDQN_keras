#!/usr/bin/env python3
import json
from argparse import ArgumentParser

import tensorflow as tf
from tqdm import tqdm

from agent import BranchingDQN
from utils import DiscreteToContinuous

parser = ArgumentParser()
args = parser.parse_args()
with open('args.txt', 'r') as f:
    args.__dict__ = json.load(f)

env = DiscreteToContinuous(args.env, args.action_per_branch)
agent = BranchingDQN(
    args.common_hidden_sizes,
    args.action_hidden_sizes,
    args.value_hidden_sizes,
    env.observation_space.shape[0],
    env.action_space.shape[0],
    args.action_per_branch,
    args)

agent.q = tf.keras.models.load_model(args.model_path)

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
