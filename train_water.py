import json
import datetime
import warnings
from argparse import ArgumentParser

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from agent import BranchingDQN
from utils import DiscreteToContinuous, epsilon_by_frame, test_fn
from cpprb import ReplayBuffer

import sys
sys.path.insert(0, "/home/anas/projects/local_bsc/perseo")
# sys.path.insert(0, "/gpfs/projects/bsc21/bsc21873/aguas/perseo")
from environments.GymEnvironments import WaterDistributionEnvironment

# Preliminary instructions
warnings.filterwarnings('ignore')

# Environment setup
ENV_ID = 'WDE-0'
INP_FILE = '/home/anas/projects/local_bsc/perseo/data/MATERNITAT.inp'
REPORT_DIR, REPORT_NAME = None, None
CONTROL_ELEMS = 'valves'
CONTROL_LIMITS = None
CONTROL_TIME_STEP = 3600
INIT_RANDOM_ACTIONS_PER_EPISODE = False

# Environment configuration
env_config = {
    'inp_file': INP_FILE,
    'list_control_elements': CONTROL_ELEMS,
    'list_action_limits': CONTROL_LIMITS,
    'time_step': CONTROL_TIME_STEP,
    'random_ini_actions': INIT_RANDOM_ACTIONS_PER_EPISODE,
} 


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--task', default=INP_FILE.split("/")[-1].split(".")[0])
    # network architecture
    parser.add_argument('--common_hidden-sizes', type=int,
                        nargs='*', default=[512, 256, 128])
    parser.add_argument('--action_hidden-sizes', type=int,
                        nargs='*', default=[64])
    parser.add_argument('--value_hidden-sizes', type=int,
                        nargs='*', default=[64])
    parser.add_argument('--action_per_branch', type=int, default=2)
    # training hyperparameters
    parser.add_argument('--epsilon_start', type=float, default=1.)
    parser.add_argument('--epsilon_final', type=float, default=0.01)
    parser.add_argument('--epsilon_decay', type=int, default=20000)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--target_net_update_freq', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_starts', type=int, default=5000)
    parser.add_argument('--max_frames', type=int, default=1000000)
    # replay buffer
    parser.add_argument('--memory_size', type=int, default=100000)
    parser.add_argument('--prioritized', type=bool, default=True)

    return parser.parse_args()

# Environment creator function


def env_creator(env_config):
    tmp_file = '{}/{}'.format(REPORT_DIR,
                              REPORT_NAME) if REPORT_NAME is not None else None
    return WaterDistributionEnvironment(tmp_file_prefix=tmp_file, **env_config)


def main(args=get_args()):
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = 'logs/' + args.task + '/' + current_time + '/train'
    summary_writer = tf.summary.create_file_writer(log_dir)
    
    env = env_creator(env_config)
    env = DiscreteToContinuous(env, args.action_per_branch)
    args.state_shape = env.observation_space.shape
    args.action_shape = env.action_space.shape
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    print("Actions per branch:", args.action_per_branch)
    env_dict = {"obs": {"shape": args.state_shape},
                "act": {"shape": args.action_shape},
                "rew": {},
                "next_obs": {"shape": args.state_shape},
                "done": {}}
    rb = ReplayBuffer(args.memory_size, env_dict)
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
                0, args.action_per_branch, size=args.action_shape[0])
        ns, r, done, _ = env.step(action)
        rb.add(obs=s, act=action, rew=r, next_obs=ns, done=done)
        ep_reward += r
        if frame > args.learning_starts:
            sample = rb.sample(args.batch_size)
            loss = agent.update_policy(sample)
        if done:
            rb.on_episode_end()
            # Run a test episode
            if step % 100 == 0:
                test_reward = test_fn(env, agent)
                with summary_writer.as_default():
                    tf.summary.scalar('test/reward', test_reward, step=int(frame/1000)) 
            ns = env.reset()
            p_bar.set_description('Rew: {:.3f}'.format(ep_reward))
            with summary_writer.as_default():
                if frame > args.learning_starts:
                    tf.summary.scalar('train/loss', loss, step=step)
                tf.summary.scalar('train/epsilon', epsilon, step=step)
                tf.summary.scalar('train/reward', ep_reward, step=step)
            step += 1
            ep_reward = 0.
        s = ns
        s = np.array(s)
        p_bar.update(1)           

    p_bar.close()

    with open('args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)


if __name__ == "__main__":
    main()
