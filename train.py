import json
import datetime
from argparse import ArgumentParser

import gym
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from agent import BranchingDQN
from utils import DiscreteToContinuous, ExperienceReplayMemory, epsilon_by_frame, test_fn


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--task', default='BipedalWalker-v3')
    # network architecture
    parser.add_argument('--common_hidden-sizes', type=int,
                        nargs='*', default=[512, 256])
    parser.add_argument('--action_hidden-sizes', type=int,
                        nargs='*', default=[128])
    parser.add_argument('--value_hidden-sizes', type=int,
                        nargs='*', default=[128])
    parser.add_argument('--action_per_branch', type=int, default=32)
    # training hyperparameters
    parser.add_argument('--epsilon_start', type=float, default=1.)
    parser.add_argument('--epsilon_final', type=float, default=0.01)
    parser.add_argument('--epsilon_decay', type=int, default=80000)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--target_net_update_freq', type=int, default=1000)
    parser.add_argument('--memory_size', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--learning_starts', type=int, default=5000)
    parser.add_argument('--max_frames', type=int, default=2000000)

    return parser.parse_args()


def main(args=get_args()):
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = 'logs/' + args.task + '/' + current_time + '/train'
    summary_writer = tf.summary.create_file_writer(log_dir)
    
    env = gym.make(args.task)
    env = DiscreteToContinuous(env, args.action_per_branch)
    args.state_shape = env.observation_space.shape
    args.action_shape = env.action_space.shape
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    print("Actions per branch:", args.action_per_branch)
    memory = ExperienceReplayMemory(args.memory_size)
    agent = BranchingDQN(
        args.common_hidden_sizes,
        args.action_hidden_sizes,
        args.value_hidden_sizes,
        args.state_shape[0],
        args.action_shape[0],
        args.action_per_branch,
        args.lr,
        args.target_net_update_freq)

    s = env.reset()
    s = np.array(s)

    ep_reward = 0.
    step = 0

    p_bar = tqdm(total=args.max_frames)
    for frame in range(args.max_frames):
        epsilon = epsilon_by_frame(frame, args.epsilon_start, args.epsilon_final, args.epsilon_decay)
        if np.random.random() > epsilon:
            action = agent.get_action(np.expand_dims(s, 0))
        else:
            action = np.random.randint(
                0, args.action_per_branch, size=env.action_space.shape[0])
        ns, r, done, _ = env.step(action)
        ep_reward += r
        if frame % 1000 == 0:
            test_reward = test_fn(env, agent)
            with summary_writer.as_default():
                tf.summary.scalar('test/reward', test_reward, step=int(frame/1000)) 
        if frame > args.learning_starts:
            loss = agent.update_policy(memory, args)
        if done:
            ns = env.reset()
            p_bar.set_description('Rew: {:.3f}'.format(ep_reward))
            with summary_writer.as_default():
                if frame > args.learning_starts:
                    tf.summary.scalar('train/loss', loss, step=step)
                tf.summary.scalar('train/epsilon', epsilon, step=step)
                tf.summary.scalar('train/reward', ep_reward, step=step)
            step += 1
            ep_reward = 0.
        memory.push((s, action, r, ns, 0. if done else 1.))
        s = ns
        s = np.array(s)
        p_bar.update(1)           

    p_bar.close()

    with open('args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)


if __name__ == "__main__":
    main()
