from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns

from pathlib import Path
from typing import Optional

sns.set()
sns.set_style("whitegrid")

class Analyzer:
    def __init__(self):
        self.episode_index = []
        self.rewards = []
        self.episode_lengths = []

    def save_episode(self, episode:int, reward:float, length:int) -> None:
        """ Save an episode.

        Args:
            episode: Index of the episode to be saved.
            reward: Total reward of the episode.
            length: Number of time steps performed in the episode.

        """
        self.episode_index.append(episode)
        self.rewards.append(reward)
        self.episode_lengths.append(length)

    def print_status(self, lag: int = 100) -> None:
        """ Print the current training status.

        Args:
            lag: Number of rewards used to compute the average reward.

        """
        episode = self.episode_index[-1]
        reward = self.rewards[-1]
        steps = self.episode_lengths[-1]
        out = (
            "Episode: {:4} - "
            "Total Reward: {:5.2f} - "
            "Avg. Reward: {:5.2f} - "
            "Steps: {:3}").format(episode,
                                  reward,
                                  self.average_reward(lag),
                                  steps)

        print(out)

    def average_reward(self, lag: int = 100, i: Optional[int] = None) -> float:
        """ Compute the rolling average reward.

        Args:
            lag: Number of elements used to compute the mean.
            i: Episode from which to compute the mean. If None the last
                observed episode is used.

        Returns:
            average_reward: Mean reward

        """
        if i is None:
            return np.mean(self.rewards[-lag:])
        else:
            return np.mean(self.rewards[max(i-lag,0):i+1])

    def plot_training(self, filename: str, lag: int = 100) -> None:
        """ Plot the training process.

        """
        path = Path(filename)

        fig, ax = plt.subplots(tight_layout=True)
        sns.set()

        ax.plot(self.episode_index, self.rewards, alpha=.8, label="Reward")
        average_rewards = [self.average_reward(lag=lag, i=i)
                           for i in range(len(self.episode_index))]

        ax.plot(self.episode_index,
                average_rewards,
                label="Avg. Reward")

        ax.set_ylabel("Reward")
        ax.set_xlabel("Episode")
        ax.legend()

        fig.savefig(path)
