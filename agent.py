#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from tensorflow import keras

import network


class BranchingDQN:
    def __init__(self, hiddens_common, hiddens_actions, hiddens_value, num_states, num_action_branches, action_per_branch, lr, target_net_update_freq):
        self.num_states = num_states
        self.num_action_branches = num_action_branches
        self.action_per_branch = action_per_branch

        self.q = network.branching_net(
            hiddens_common, hiddens_actions, hiddens_value, num_states, self.num_action_branches, action_per_branch)
        self.target = network.branching_net(
            hiddens_common, hiddens_actions, hiddens_value, num_states, self.num_action_branches, action_per_branch)

        # Init the target network with the same weights as the main network
        weights = self.q.get_weights()
        self.target.set_weights(weights)

        self.optimizer = keras.optimizers.Adam(
            learning_rate=lr, clipnorm=1)

        self.target_net_update_freq = target_net_update_freq
        self.update_counter = 0

    @staticmethod
    def loss_function(x, y):
        return tf.reduce_mean(tf.reduce_sum(tf.multiply(x-y, x-y), axis=-1))

    def get_action(self, x):
        out = self.q(x, training=False)
        action = np.argmax(out, axis=2)[0]
        return action

    @tf.function
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
        self.optimizer.apply_gradients(zip(grads, self.q.trainable_variables))
        return loss

    def update_policy(self, memory, params):

        b_states, b_actions, b_rewards, b_next_states, b_masks = memory.sample(
            params.batch_size)

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
        mean_max_next_q_vals = tf.reduce_mean(
            tf.reduce_sum(max_next_q_vals, axis=-1), axis=-1)

        target_qvals = rewards + mean_max_next_q_vals*params.gamma*masks
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
            
        return loss
