import datetime
from utils import DiscreteToContinuous, epsilon_by_frame, test_fn
from agent import BranchingDQN

from tqdm import tqdm

import json
import os
import time
import warnings
from argparse import ArgumentParser
from multiprocessing import Event, Process, SimpleQueue, shared_memory
import multiprocessing

import gym
import mlflow
import numpy as np
from cpprb import (MPPrioritizedReplayBuffer, ReplayBuffer)

warnings.filterwarnings('ignore')
# Number of explorers
NUM_PROC = 2
# Epsilon
EPSILON = 1.0

class ExplorerBDQ:
    def __init__(self, hiddens_common, hiddens_actions, hiddens_value, state_shape, num_action_branches, action_per_branch):
        self.state_shape = state_shape
        self.num_action_branches = num_action_branches
        self.action_per_branch = action_per_branch
        import network
        self.q = network.branching_net(
            hiddens_common, hiddens_actions, hiddens_value, state_shape, self.num_action_branches, action_per_branch)

    def get_action(self, x: np.ndarray, epsilon=0) -> np.ndarray:
        batch_size = 1 if len(x.shape) == 1 else x.shape[0]
        if np.random.random() < epsilon:
            action = np.random.randint(
                0, self.action_per_branch, size=(batch_size, self.num_action_branches))
        else:
            if batch_size == 1:
                x = np.expand_dims(x, axis=0)
            out = self.q(x, training=False)
            action = np.argmax(out, axis=-1)
        if np.random.random() < (2*epsilon if (2*epsilon < 1) else 1):
            action[np.random.randint(batch_size), np.random.randint(
                self.num_action_branches)] = np.random.randint(self.action_per_branch)
        return action.squeeze()

    @property
    def weights(self):
        return self.q.get_weights()

    @weights.setter
    def weights(self, w):
        self.q.set_weights(w)
        
def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())
    print(multiprocessing.current_process()._identity)
    

def explorer(global_rb, is_training_done, queue, args, shm_name):
    info('Explorer')
    env = env_creator(args)
    env_dict = {"obs": {"shape": args.state_shape},
                "act": {"shape": args.action_shape},
                "rew": {},
                "next_obs": {"shape": args.state_shape},
                "done": {}}
    proc_id = multiprocessing.current_process()._identity[0] - 1
    local_buffer_size = 96
    local_rb = ReplayBuffer(local_buffer_size, env_dict)

    try:
        model = ExplorerBDQ(
            args.common_hidden_sizes,
            args.action_hidden_sizes,
            args.value_hidden_sizes,
            args.state_shape,
            args.action_shape[0],
            args.action_per_branch
        )
    except Exception as e:
        print(e)
        print('Explorer: Failed to create model')
        is_training_done.set()
        return

    obs = env.reset()
    ep_reward = 0.
    while not is_training_done.is_set():
        if not queue.empty():
            w = queue.get()
            model.weights = w

        action = model.get_action(obs, EPSILON)
        next_obs, reward, done, _ = env.step(action)
        ep_reward += reward
        local_rb.add(obs=obs, act=action, rew=reward,
                     next_obs=next_obs, done=done)

        if done:
            local_rb.on_episode_end()
            obs = env.reset()
            dummy_array = np.zeros(NUM_PROC)
            existing_shm = shared_memory.SharedMemory(name=shm_name)
            episode_rewards = np.ndarray(dummy_array.shape, dtype=dummy_array.dtype, buffer=existing_shm.buf) # Attach to the existing shared memory block
            episode_rewards[proc_id] = ep_reward
            ep_reward = 0.
        else:
            obs = next_obs

        if local_rb.get_stored_size() == local_buffer_size:
            local_sample = local_rb.get_all_transitions()
            local_rb.clear()

            global_rb.add(**local_sample)


def learner(global_rb, queues, args, shm_name):
    import tensorflow as tf
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = 'logs/' + args.task + '/' + current_time
    summary_writer = tf.summary.create_file_writer(log_dir)
    
    batch_size = args.batch_size
    n_warmup = args.learning_starts
    n_training_step = args.max_frames
    explorer_update_freq = 100

    env = env_creator(args)

    model = BranchingDQN(
        args.common_hidden_sizes,
        args.action_hidden_sizes,
        args.value_hidden_sizes,
        args.state_shape,
        args.action_shape[0],
        args.action_per_branch,
        args.lr,
        args.target_net_update_freq)

    while global_rb.get_stored_size() < n_warmup:
        # make a progress bar with global_rb.get_stored_size() and args.learning_starts
        print (global_rb.get_stored_size())
        print("Waiting for explorers to fill replay buffer")	
        time.sleep(1)

    p_bar = tqdm(total=args.max_frames)
    for frame in tqdm(range(n_training_step)):
        global EPSILON
        EPSILON = epsilon_by_frame(
            frame - args.learning_starts, args.epsilon_start, args.epsilon_final, args.epsilon_decay)

        sample = global_rb.sample(batch_size)

        loss, absTD = model.update_policy(sample)
        global_rb.update_priorities(sample["indexes"], absTD)

        if frame % explorer_update_freq == 0:
            w = model.weights
            for q in queues:
                q.put(w)

        # Run a test episode
        if frame % 1000 == 0:
            test_reward = test_fn(env, model)
            mlflow.log_metric("test reward", test_reward)
        dummy_array = np.zeros(NUM_PROC)
        existing_shm = shared_memory.SharedMemory(name=shm_name)
        episode_rewards = np.ndarray(dummy_array.shape, dtype=dummy_array.dtype, buffer=existing_shm.buf) # Attach to the existing shared memory block
        p_bar.set_description('Rew: {:.3f}'.format(np.mean(episode_rewards)))
        
        # log the metrics with tensorboard
        with summary_writer.as_default():
            tf.summary.scalar('train/loss', loss, frame)
            tf.summary.scalar('train/td_error', np.mean(absTD), frame)
            tf.summary.scalar('train/epsilon', EPSILON, frame)
            tf.summary.scalar('train/reward', np.mean(episode_rewards), frame)
            tf.summary.scalar('test/reward', test_reward, frame)
        
        p_bar.update(1)

    p_bar.close()

def env_creator(args):
    env = gym.make(args.task)
    env = DiscreteToContinuous(env, args.action_per_branch)
    return env

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
    parser.add_argument('--action_per_branch', type=int, default=6)
    # training hyperparameters
    parser.add_argument('--epsilon_start', type=float, default=1.0)
    parser.add_argument('--epsilon_final', type=float, default=0.01)
    parser.add_argument('--epsilon_decay', type=int, default=20000)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--target_net_update_freq', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_starts', type=int, default=192)
    parser.add_argument('--max_frames', type=int, default=1000000)
    # replay buffer
    parser.add_argument('--buffer_size', type=int, default=100000)
    parser.add_argument('--prioritized', type=bool, default=True)
    # tracking
    parser.add_argument('--tracking_uri', type=str, default='http://0.0.0.0:5000')

    return parser.parse_args()


def main(args=get_args()):
    # mlflow tracking
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.task)
    # mlflow log args as params
    for a in vars(args):
       mlflow.log_param(a, getattr(args, a)) 
    # Shared memory numpy array for episode rewards from explorers
    dummy_array = np.zeros(NUM_PROC)
    shm = shared_memory.SharedMemory(create=True, size=dummy_array.nbytes)
    try:
        env = env_creator(args)
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
        n_explorer = NUM_PROC
        global_rb = MPPrioritizedReplayBuffer(args.buffer_size, env_dict)

        is_training_done = Event()
        is_training_done.clear()

        qs = [SimpleQueue() for _ in range(n_explorer)]
        ps = [Process(target=explorer,
                    args=[global_rb, is_training_done, q, args, shm.name])
            for q in qs]

        for p in ps:
            p.start()

        learner(global_rb, qs, args, shm.name)
        is_training_done.set()

        for p in ps:
            p.join()

        print(global_rb.get_stored_size())
        shm.close()
        shm.unlink()
        
    except Exception as e:
        print(e)
        shm.close()
        shm.unlink()
        raise e

    with open('args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)


if __name__ == "__main__":
    main()