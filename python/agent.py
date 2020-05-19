import dqn
import gym
import random
import numpy as np

class Agent:
    def __init__(self,
                 observation_space: gym.spaces.box.Box,
                 action_space: gym.spaces.discrete.Discrete,
                 ϵ: float = 1.0,
                 γ: float = 0.99,
                 C: int = 1000):
        """
        Args:
            observation_space: Observation space object
            action_space: Action space object
            ϵ: Initial exploration rate
            γ: Discount factor
            C: Target update frequency for double Q-learning
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.ϵ0 = ϵ
        self.ϵ = self.ϵ0
        self.γ = γ
        self.C = C

        # Keep separate online and target networks (see DDQN paper).
        self.Q_online = dqn.DQN(self.observation_space.shape, self.action_space.n)
        self.Q_target = dqn.DQN(self.observation_space.shape, self.action_space.n)
        self.Q_target.copy_weights(self.Q_online)

        self.update_counter = 0

    def get_action(self, state:np.ndarray) -> int:
        """ Given a state s, returns the action under the current policy.

        Args:
            state: Observation of a state.

        Returns:
            action: Random action w/probability ϵ, else the greedy action
                argmax_a' Q(s,a') where Q denotes the online DQN.
        """
        if random.random() < self.ϵ:
            return self.action_space.sample()

        else:
            state = state.reshape(1,-1) # Make observation shape compatible
            Qs = self.Q_online.forward(state)
            return np.argmax(Qs)

    def update(self, minibatch:list, episode:int) -> None:
        """ Update the agent given a minibatch of experiences. Performs one
        pass over the minibatch to update the online network, using the target
        network to compute the targets. Anneals ϵ linearly, and updates the
        target network every 1000 times the online network is update.

        Args:
            minibatch: List of (s, a, r, s', done) samples.
            episode: Current episode.

        """
        B = len(minibatch)
        states = np.zeros(shape=(len(minibatch), self.observation_space.shape[0]))
        next_states = np.zeros_like(states)
        targets = np.zeros(len(minibatch))
        rewards = np.zeros_like(targets)
        d = np.zeros_like(targets, dtype=np.bool)
        actions = np.zeros_like(targets, dtype=np.int)
        for i, (s, a, r, s_, terminal) in enumerate(minibatch):
            states[i,:] = s
            next_states[i,:] = s_
            actions[i] = a
            d[i] = 1 - terminal
            rewards[i] = r

        Q = self.Q_target.model.predict(states)
        Q_next = self.Q_target.model.predict(next_states)

        target = Q.copy() # Easier to understand name

        target[np.arange(B), actions] = rewards \
                                      + self.γ * np.max(Q_next, axis=1) * d

        self.Q_online.update(states, target)

        self.update_ϵ(episode)
        self.update_counter += 1
        if self.update_counter % self.C == 0:
            self.Q_target.copy_weights(self.Q_online)

    def update_ϵ(self, episode:int) -> None:
        decay_until = 1000
        self.ϵ = max((self.ϵ0 * (1 - episode/decay_until), 0.01))
