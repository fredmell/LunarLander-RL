import dqn
import gym
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from agent import Agent
from replay_buffer import ReplayBuffer
from analyzer import Analyzer

def main(M,
         T,
         minibatch_size=64,
         train_after=20,
         goal=250.0,
         lag=100,
         show=True):
    """ Run M episodes of Q-learning on the discrete Lunar Lander environment.

    Args:
        M (int) : Max number of episodes.
        T (int) : Max episode length.
        minibatch_size (int) : Size of minibatch of experiences to replay.
            Defaults to 64, ideally use powers of 2 (for optimal GPU usage).
        train_after (int) : Number of episodes to run before training the Q net.
        goal (float) : Average score for having 'solved' the game. Defaults to
            250.0, for which a successful landing needs to have occured.
        lag (int) : Number of episodes used to compute the average score.
        show(bool) : Indicates whether to show the episode window or not.

    """
    total_rewards = []

    env = gym.make("LunarLander-v2")
    env.reset()

    agent = Agent(env.observation_space, env.action_space)
    buffer = ReplayBuffer()
    analyzer = Analyzer()

    for episode in range(M):
        state = env.reset()
        rewards = []
        for t in range(T):
            # Interact with the environment
            env.render()
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)

            # Store the experience, sample experiences
            buffer.store(state, action, reward, next_state, done)

            # Sample from the experiences and use these to update the Q-network
            if episode > train_after:
                minibatch = buffer.sample(minibatch_size)
                agent.update(minibatch, episode)

            state = next_state
            rewards.append(reward)

            # End the episode if it is over.
            if done:
                break

        analyzer.save_episode(episode, sum(rewards), t)
        analyzer.print_status(lag=lag)

        if analyzer.average_reward(lag=lag) > goal:
            print("Game solved!")
            break

    env.close()
    analyzer.save_report()

if __name__ == '__main__':
    M = 1500 # Max number of episodes
    T = 400  # Max number of steps per episode
    main(M, T)
