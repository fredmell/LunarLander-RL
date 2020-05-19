import numpy as np
import random

from typing import Iterable

class ReplayBuffer:
    def __init__(self, N: int = 100000):
        """ Create a replay buffer object, essentially wraps a list of
        (s,a,r,s') tuples.

        Args:
            N: Size of the replay buffer

        """
        self.N = N
        self.buffer = []

    def __len__(self) -> int:
        return len(self.buffer)

    def store(self,
              state: np.ndarray,
              action: int,
              reward: float,
              next_state: np.ndarray,
              terminal: bool
              ) -> None:
        """ Store a (s,a,r,s') tuple in the buffer. Removes the oldest
        observation if the buffer is full.

        Args:
            state: Observation at the current state.
            action: Integer indicating the action taken.
            reward: Reward of (state,action,next state) triplet.
            next_state: Observation at the next state.
            terminal: Indicates if the episode ended from the transition.

        """
        transition = (state, action, reward, next_state, terminal)

        if len(self.buffer) == self.N:
            del self.buffer[0]

        self.buffer.append(transition)

    def sample(self, n:int) -> Iterable:
        """ Sample a random minibatch of experiences from the buffer, with
        uniform probability.

        Args:
            n: Minibatch size (number of transitions).

        Returns:
            batch: Minibatch of n experience tuples.

        """
        n = min(len(self.buffer), n)
        batch = random.sample(self.buffer, n)
        return batch
