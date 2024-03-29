# Branching NETWORK
################################
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def branching_net(hiddens_common, hiddens_actions, hiddens_value, state_shape, num_action_branches, action_per_branch):

    state_input = layers.Input(state_shape, name='Input')
    out = state_input

    # Create the shared network module "Common State-Value Estimator"
    for i, hidden in enumerate(hiddens_common):
        if hidden != 0:
            out = layers.Dense(hidden, activation="relu",
                               name='common{}'.format(i))(out)

    # Create the action branches, one for each action dimension
    action_scores = []
    for action_stream in range(num_action_branches):
        action_out = out
        for hidden in hiddens_actions:
            if hidden != 0:
                action_out = layers.Dense(
                    hidden, activation="relu", name='action{}'.format(action_stream))(action_out)
        # Output layer for the action branch, outputs A_d (Advantage values for actions in dimension d)
        action_scores.append(layers.Dense(action_per_branch, activation=None,
                             name='action_bin{}'.format(action_stream))(action_out))

    total_action_scores = tf.stack(action_scores, axis=1)
    mean_action_scores = tf.reduce_mean(total_action_scores, 2, keepdims=True)
    adj_action_scores = tf.subtract(total_action_scores, mean_action_scores)

    # Create the state value branch
    state_out = out
    for i, hidden in enumerate(hiddens_value):
        if hidden != 0:
            state_out = layers.Dense(
                hidden, activation="relu", name='state{}'.format(i))(state_out)
    # Output layer outputs the state score
    state_score = layers.Dense(1, activation=None, name='state_out')(state_out)
    state_score = tf.expand_dims(state_score, 1)

    q_values = tf.add(state_score, adj_action_scores)

    # Take state as input and outputs Q-values
    model = keras.Model(state_input, q_values)

    return model


def plot():
    model = branching_net([512, 256], [128], [128], 9, 4, 6)
    print(model.summary())
    keras.utils.plot_model(model, show_shapes=True)


if __name__ == "__main__":
    plot()
