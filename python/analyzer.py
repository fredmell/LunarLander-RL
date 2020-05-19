from __future__ import annotations

import numpy as np
import pickle

class Analyzer:
    def __init__(self):
        self.episode_index = []
        self.rewards = []
        self.episode_lengths = []

    def save_episode(self, episode:int, reward:float, length:int) -> None:
        self.episode_index.append(episode)
        self.rewards.append(reward)
        self.episode_lengths.append(length)

    def print_status(self, lag: int = 100) -> None:
        episode = self.episode_index[-1]
        reward = self.rewards[-1]
        steps = self.episode_lengths[-1]
        out = (
            "Episode: {:4} - "
            "Total Reward: {:5.2f} - "
            "Avg. Reward: {:5.2f} - "
            "Steps: {:3}").format(episode, reward, self.average_reward(lag), steps)

        print(out)

    def average_reward(self, lag: int = 100) -> float:
        return np.mean(self.rewards[-lag:])

    def save_report(self, filename: str = "analysis.pkl") -> None:
        with open(filename, "wb") as outfile:
            pickle.dump(self, outfile)

    def load_report(self, filename: str) -> Analyzer:
        with open(filename, "rb") as infile:
            return pickle.load(infile)
