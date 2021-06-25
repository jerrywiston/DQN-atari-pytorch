import time
import gym

# Build Environment
env_name = "PongNoFrameskip-v4"
env = gym.make(env_name)

print("environment:", env_name)
print("action space:", env.action_space.n)
print("action:", env.unwrapped.get_action_meanings())
print("observation space:", env.observation_space.shape)

# Start Environment
state = env.reset()
while True:
    action = env.action_space.sample()
    state_next, reward, done, info = env.step(action)
    env.render()
    time.sleep(0.01)
    if done:
        break
