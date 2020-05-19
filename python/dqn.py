from __future__ import annotations

import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from typing import Iterable

class DQN:
    def __init__(self, obs_shape, n_actions, learning_rate: float = 0.0002):
        self.obs_shape = obs_shape
        self.n_actions = n_actions

        self.lr = learning_rate

        self.model = self.generate_model()

    def generate_model(self,
                       layer_sizes: Iterable = (256, 128),
                       activation: str = 'relu',
                       ) -> tf.keras.Sequential:
        """ Generate the Keras NN.

        Args:
            layer_sizes : Sizes of hidden layers.
            activation  : Activation to use for all hidden layers.

        Returns:
            model : The tf.keras Q network model

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
            state : State observations with shape (n_obs, self.obs_shape).

        Returns:
            Qs : Estimated Q values, one for each action, shape
                (n_obs, self.n_actions).

        """
        Qs = self.model.predict(state)
        return Qs

    def update(self, states: np.ndarray, targets: np.ndarray) -> None:
        """ Wrapper for calling fit, with input states and target targets.
        Performs one pass through the training set.

        Args:
            states  : State observations.
            targets : Training targets.

        """
        self.model.fit(states, targets, epochs=1, verbose=0)

    def copy_weights(self, other: DQN) -> None:
        """ Set weights to a copy of those of otherDQN.

        Args:
            otherDQN : Q-network whose weights we copy and assign to self.

        """
        self.model.set_weights(other.model.get_weights())
