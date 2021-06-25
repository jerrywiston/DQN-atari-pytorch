import gym
import numpy as np
from PIL import Image

class PongEnvWrapper(gym.Wrapper):
    def __init__(self, env, k, img_size=(84,84)):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.img_size = img_size
        obs_shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(k, img_size[0], img_size[1]), dtype=np.float32)

    def _preprocess(self, state, th=0.4):
        state = np.array(Image.fromarray(state).resize(self.img_size,Image.BILINEAR))
        state = state.astype(np.float).mean(2) / 255.
        state[state > th] = 1.0
        state[state <=th] = 0.0
        return state

    def reset(self):
        state = self.env.reset()
        state = self._preprocess(state)
        state = state[np.newaxis, ...].repeat(self.k, axis=0)
        return state

    def step(self, action):
        state_next = []
        info = []
        reward = 0
        done = False
        for i in range(self.k):
            if not done:
                state_next_f, reward_f, done_f, info_f = self.env.step(action)
                state_next_f = self._preprocess(state_next_f)
                reward += reward_f
                done = done_f
                info.append(info_f)
            state_next.append(state_next_f[np.newaxis, ...])
        state_next = np.concatenate(state_next, 0)

        return state_next, reward, done, info

if __name__ == "__main__":
    import time
    import cv2

    # Build Environment
    env_name = "PongNoFrameskip-v4"
    env_ = gym.make(env_name)
    env = PongEnvWrapper(env_, k=4, img_size=(84,84))

    print("environment:", env_name)
    print("action space:", env.action_space.n)
    print("action:", env.unwrapped.get_action_meanings())
    print("observation space:", env.observation_space.shape)

    # Start Environment
    state = env.reset()
    cv2.imshow("pong", state[0])
    cv2.waitKey(0)
    while True:
        action = env.action_space.sample()
        state_next, reward, done, info = env.step(action)
        env.render()
        cv2.imshow("pong", state_next[0])
        cv2.waitKey(10)
        if done:
            break
