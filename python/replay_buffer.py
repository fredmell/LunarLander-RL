import random

class ReplayBuffer:
    def __init__(self, N=100000):
        """ Create a replay buffer object, essentially wraps a list of
        (s,a,r,s') tuples.

        Args:
            N (int): Size of the replay buffer

        """
        self.N = N
        self.buffer = []

    def __len__(self):
        return len(self.buffer)

    def store(self, s, a, r, s_, terminal):
        """ Store a (s,a,r,s') tuple in the buffer. Removes the oldest
        observation if the buffer is full.

        Args:
            s (np.array) : Observation at the current state.
            a (int) : Integer indicating the action taken.
            r (int) : Reward of (s,a) pair.
            s_ (np.array) : Observation at the next state, after taking action
                            a at state s.
            terminal (bool) : Indicates if the state ended the episode.

        """
        transition = (s, a, r, s_, terminal)

        if len(self.buffer) == self.N:
            del self.buffer[0]

        self.buffer.append(transition)

    def sample(self, n):
        """ Sample a random minibatch of experiences from the buffer, with
        uniform probability.

        Args:
            n (int) : Minibatch size (number of transitions).

        Returns:
            batch (list) : Minibatch of n experience tuples.

        """
        n = min(len(self.buffer), n)
        batch = random.sample(self.buffer, n)
        return batch
