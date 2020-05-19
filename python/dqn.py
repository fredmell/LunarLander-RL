import numpy as np
import tensorflow as tf

from tensorflow.keras import layers

class DQN:
    def __init__(self, obs_shape, n_actions, learning_rate=0.0002):
        self.obs_shape = obs_shape
        self.n_actions = n_actions

        self.lr = learning_rate

        self.model = self.generate_model()

    def generate_model(self, layer_sizes=(256, 128), activation='relu'):
        """ Generate the Keras NN.

        Args:
            layer_sizes (iterable): Sizes of hidden layers.
            activation (string) : Activation to use for all hidden layers.

        Returns:
            model (tf.keras.Sequential) : The tf.keras Q network model

        """
        model = tf.keras.Sequential()

        model.add(tf.keras.Input(shape=self.obs_shape))
        for l, size in enumerate(layer_sizes):
            model.add(layers.Dense(size, activation=activation))
            model.add(layers.Dense(size, activation=activation))
        model.add(layers.Dense(self.n_actions, activation='linear'))

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
            loss='mse'
        )

        return model

    def forward(self, state):
        """ Wrapper for keras predict, feeds input forward through the network.

        Args:
            state (np.ndarray) : State observations with shape
                (n_obs, self.obs_shape).

        Returns:
            Qs (np.ndarray) : Estimated Q values, one for each action, with
                shape (n_obs, self.n_actions).

        """
        Qs = self.model.predict(state)
        return Qs

    def update(self, states, targets):
        """ Wrapper for calling fit, with input states and target targets.
        Performs one pass through the training set.

        Args:
            states (np.ndarray) : State observations.
            targets (np.ndarray) : Training targets.

        """
        self.model.fit(states, targets, epochs=1, verbose=0)

    def copy_weights(self, otherDQN):
        """ Set weights to a copy of those of otherDQN.

        Args:
            otherDQN (DQN) : Q-network whose weights we copy and assign to self.

        """
        self.model.set_weights(otherDQN.model.get_weights())
