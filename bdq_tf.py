#!/usr/bin/env python3
from tqdm import tqdm
import tensorflow as tf
import numpy as np
from tensorflow import keras
from utils import ExperienceReplayMemory, AgentConfig, BranchingTensorEnv_tf
import utils
import bdq_network

class BranchingDQN():

    def __init__(self, hiddens_common, hiddens_actions, hiddens_value, num_states, num_action_branches, action_per_branch, config):
        self.num_states = num_states
        self.num_action_branches = num_action_branches
        self.action_per_branch = action_per_branch

        self.q = bdq_network.branching_net(hiddens_common, hiddens_actions, hiddens_value, num_states, self.num_action_branches, action_per_branch)
        self.target = bdq_network.branching_net(hiddens_common, hiddens_actions, hiddens_value, num_states, self.num_action_branches, action_per_branch)
        
#         self.q = keras.models.load_model('./')
#         self.target = keras.models.load_model('./')
        
        # Init the target network with the same weights as the main network
        weights = self.q.get_weights()
        self.target.set_weights(weights)

        self.loss_function = keras.losses.MSE
        self.optimizer = keras.optimizers.Adam(learning_rate=config.lr, clipnorm=1)

        self.target_net_update_freq = config.target_net_update_freq
        self.update_counter = 0

    def get_action(self, x):
        out = self.q(x, training=False)
        action = np.argmax(out, axis=2)[0]
        return action
    
    # @tf.function
    def train_step(self, states, action_mask, target_qvals):
        with tf.GradientTape() as tape:
            # Train the model on the states and updated Q-values
            q_values = self.q(states)

            # Apply the masks to the Q-values to get the Q-value for action taken
            q_action = tf.multiply(q_values, action_mask)

            # Calculate loss between target Q-value and old Q-value
            loss = self.loss_function(target_qvals, q_action)

        # Backpropagation
        grads = tape.gradient(loss, self.q.trainable_variables)
        # print(grads[-2:])
        self.optimizer.apply_gradients(zip(grads, self.q.trainable_variables))
        return loss
        

    def update_policy(self, memory, params):

        b_states, b_actions, b_rewards, b_next_states, b_masks = memory.sample(params.batch_size)

        states = np.array(b_states).squeeze()
        next_states = np.array(b_next_states).squeeze()
        actions = b_actions
        rewards = b_rewards
        masks = tf.convert_to_tensor(b_masks)

        next_q_values = self.q(next_states, training=False)
        argmax_a = tf.math.argmax(next_q_values, axis=-1)

        target_next = self.target(next_states, training=False)

        argmax_a_mask = tf.one_hot(argmax_a, self.action_per_branch)
        max_next_q_vals = tf.multiply(target_next, argmax_a_mask)
        mean_max_next_q_vals = tf.reduce_mean(tf.reduce_sum(max_next_q_vals, axis=-1), axis=-1)

        target_qvals = rewards + mean_max_next_q_vals*0.99*masks
        target_qvals = tf.reshape(tf.repeat(target_qvals, self.num_action_branches * self.action_per_branch), 
                                  (-1, self.num_action_branches, self.action_per_branch))
        

        action_mask = tf.one_hot(actions, self.action_per_branch)
        target_qvals = tf.multiply(target_qvals, action_mask)
        
        loss = self.train_step(states, action_mask, target_qvals)

        self.update_counter += 1
        if self.update_counter % self.target_net_update_freq == 0:
            self.update_counter = 0
            weights = self.q.get_weights()
            self.target.set_weights(weights)


def main():
    bins = 5
    
    env = BranchingTensorEnv_tf('BipedalWalker-v3', bins)
    
    config = AgentConfig(batch_size=5, learning_starts=5)
    memory = ExperienceReplayMemory(config.memory_size)
    agent = BranchingDQN([8, 8], [4], [4], env.observation_space.shape[0], env.action_space.shape[0], bins, config)
    
    s = env.reset()
    s = np.array(s)
    
    ep_reward = 0.
    recap = []

    p_bar = tqdm(total = config.max_frames)
    for frame in range(config.max_frames):
        
        epsilon = config.epsilon_by_frame(frame)
        
        if np.random.random() > epsilon:
            action = agent.get_action(s)
        else:
            action = np.random.randint(0, bins, size = env.action_space.shape[0])
            
        ns, r, done, infos = env.step(action)
        ep_reward += r
            
        if done:
            ns = env.reset()
            recap.append(ep_reward)
            p_bar.set_description('Rew: {:.3f}'.format(ep_reward))
            ep_reward = 0.
                
        memory.push((s, action, r, ns, 0. if done else 1.))
        s = ns
        s = np.array(s)
                
        p_bar.update(1)

        if frame > config.learning_starts:
            agent.update_policy(memory, config)
                    
        if frame % 1000 == 0:
            utils.save_tf(agent, recap, 'BipedalWalker-v3')
                        
    p_bar.close()

if __name__ =="__main__":
    main()
