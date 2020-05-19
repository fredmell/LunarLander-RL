import numpy as np
import tensorflow as tf

from tensorflow.keras import layers

class DQN:
    def __init__(self, obs_shape, n_actions):
        self.obs_shape = obs_shape
        self.n_actions = n_actions

        self.lr = 0.0002

        self.model = self.generate_model()

    def generate_model(self):
        """ Generate the Keras NN.

        """
        model = tf.keras.Sequential()

        model.add(tf.keras.Input(shape=self.obs_shape))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(self.n_actions, activation='linear'))

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
            loss='mse'
        )

        return model

    def forward(self, state):
        """ Wrapper for keras predict.

        """
        Qs = self.model.predict(state)
        return Qs

    def update(self, states, targets):
        """ Wrapper for calling fit, with input states and target targets.

        """
        self.model.fit(states, targets, epochs=1, verbose=0)

    def copy_weights(self, otherDQN):
        self.model.set_weights(otherDQN.model.get_weights())
