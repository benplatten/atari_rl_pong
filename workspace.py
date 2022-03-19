import gym

env = gym.make("PongNoFrameskip-v4")
env.reset()
for _ in range(300):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()


