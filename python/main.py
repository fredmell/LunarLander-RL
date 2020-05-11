import gym

T = 100

env = gym.make("LunarLander-v2")
env.reset()

done = False
for t in range(T):
    env.render()

    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)

    if done:
        print(f"Episode finished after {t} timesteps")
        break

env.close()
