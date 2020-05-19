import dqn
import gym
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from agent import Agent
from replay_buffer import ReplayBuffer

T = 400 # Max number of steps per episode
M = 2000  #
minibatch_size = 64
total_rewards = []

env = gym.make("LunarLander-v2")
env.reset()

agent = Agent(env.observation_space, env.action_space)
buffer = ReplayBuffer()

done = False
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

        if episode > 0:
            # Sample a minibatch and update the Q-network
            minibatch = buffer.sample(minibatch_size)
            agent.update(minibatch, episode)

        state = next_state
        rewards.append(reward)
        if done:
            break

    total_rewards.append(sum(rewards))
    rolling_mean = np.mean(total_rewards[-100:])

    if rolling_mean > 200:
        print("Game solved!")
        break

    print("Episode {} - Total Reward {:.3f} - Avg. Reward {:.3f} - Buffer size {} - Episode steps {}".format(episode,
                                                                 sum(rewards),
                                                                 rolling_mean,
                                                                 len(buffer),
                                                                 t+1))


env.close()

rolling_mean_reward = [np.mean(total_rewards[(i - min(100,i)):i+1]) for i in len(total_rewards)]
plt.plot(rolling_mean_reward)
plt.show()
