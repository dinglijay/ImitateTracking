import gym
import track_env
env =gym.make('track-v0')
print(env.action_space)
print(env.action_space.shape)
print(env.observation_space)

ob = env.reset()
#env.render()
env.step([0.01,0.01,0.01])
while True:
    _, reward, done, _ = env.step([0,-0.01,0])
    print(reward)
    if done:
        break
