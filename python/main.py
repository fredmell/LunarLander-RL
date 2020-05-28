import dqn
import gym
import numpy as np

from agent import Agent
from replay_buffer import ReplayBuffer
from analyzer import Analyzer

def main(M: int,
         T: int,
         minibatch_size: int = 64,
         train_after: int = 20,
         goal: float = 250.0,
         lag: int = 100,
         show: bool = True
         ) -> None:
    """ Run M episodes of Q-learning on the discrete Lunar Lander environment.

    Args:
        M              : Max number of episodes.
        T              : Max episode length.
        minibatch_size : Size of minibatch of experiences to replay. Ideally
            use powers of 2 (for optimal GPU usage).
        train_after    : Number of episodes to run before training the Q net.
        goal           : Average score for having 'solved' the game.
        lag            : Number of episodes used to compute the average score.
        show           : Indicates whether to show the episode window or not.

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
    analyzer.plot_training()

if __name__ == '__main__':
    M = 10 # Max number of episodes
    T = 400  # Max number of steps per episode
    main(M, T)
