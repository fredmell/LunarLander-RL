import dqn
import gym
import numpy as np

from agent import Agent
from replay_buffer import ReplayBuffer
from analyzer import Analyzer

def run(M: int,
        T: int,
        minibatch_size: int = 64,
        train_after: int = 20,
        goal: float = 250.0,
        lag: int = 100,
        learning_rate: float = 0.0002,
        optimizer: str = "Adam",
        show: bool = True
        ) -> Analyzer:
    """ Run M episodes of Q-learning on the discrete Lunar Lander environment.

    Args:
        M              : Max number of episodes.
        T              : Max episode length.
        minibatch_size : Size of minibatch of experiences to replay. Ideally
            use powers of 2 (for optimal GPU usage).
        train_after    : Number of episodes to run before training the Q net.
        goal           : Average score for having 'solved' the game.
        lag            : Number of episodes used to compute the average score.
        learning_rate  : Learning rate for optimizer
        show           : Indicates whether to show the episode window or not.

    Returns:
        analyzer: Analysis object with reward values, and methods for plotting.

    """
    total_rewards = []

    env = gym.make("LunarLander-v2")
    env.reset()

    agent = Agent(
        env.observation_space,
        env.action_space,
        η=learning_rate,
        optimizer=optimizer
    )
    buffer = ReplayBuffer()
    analyzer = Analyzer()

    for episode in range(M):
        state = env.reset()
        rewards = []
        for t in range(T):
            # Interact with the environment
            if show:
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
    return analyzer
