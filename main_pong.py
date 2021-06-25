import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import gym
from env_wrapper import PongEnvWrapper
import dqn
import torch
import models
import argparse
import time
import os

def epsilon_compute(frame_id, epsilon_max=1, epsilon_min=0.05, epsilon_decay=100000):
    return epsilon_min + (epsilon_max - epsilon_min) * np.exp(-frame_id / epsilon_decay)

def save_gif(img_buffer, fname, gif_path="gif"):
    if not os.path.exists(gif_path):
        os.makedirs(gif_path)
    img_buffer[0].save(os.path.join(gif_path, fname), save_all=True, append_images=img_buffer[1:], duration=1, loop=1)

def train(env, agent, stack_frames, img_size, save_path="save", max_steps=1000000):
    total_step = 0
    episode = 0
    while True:
        # Reset environment.
        state = env.reset()

        # Initialize information.
        step = 0
        total_reward = 0
        loss = 0.

        # One episode.
        while True:
            # Select action.
            epsilon = epsilon_compute(total_step)
            action = agent.choose_action(state, epsilon)

            # Get next stacked state.
            state_next, reward, done, info = env.step(action)

            # Store transition and learn.
            total_reward += reward
            agent.store_transition(state, action, reward, state_next, done)
            if total_step > 4*agent.batch_size:
                loss = agent.learn()

            state = state_next.copy()
            step += 1
            total_step += 1

            if total_step % 100 == 0 or done:
                print('\rEpisode: {:3d} | Step: {:3d} / {:3d} | Reward: {:.3f} / {:.3f} | Loss: {:.3f} | Epsilon: {:.3f}'\
                    .format(episode, step, total_step, reward, total_reward, loss, epsilon), end="")
            
            if total_step % 10000 == 0:
                print("\nSave Model ...")
                agent.save_load_model(op="save", path=save_path, fname="qnet.pt")
                print("Generate GIF ...")
                img_buffer = play(env, agent, stack_frames, img_size)
                save_gif(img_buffer, "train_" + str(total_step).zfill(6) + ".gif")
                print("Done !!")

            if done or step>2000:
                episode += 1
                print()
                break
        
        if total_step > max_steps:
            break

def play(env, agent, stack_frames, img_size, render=False):
    # Reset environment.
    state = env.reset()
    img_buffer = [Image.fromarray(state[0]*255)]

    # Initialize information.
    step = 0
    total_reward = 0
    loss = 0.

    # One episode.
    while True:
        # Select action.
        action = agent.choose_action(state, 0)

        # Get next stacked state.
        state_next, reward, done, info = env.step(action)
        if step % 2 == 0:
            img_buffer.append(Image.fromarray(state_next[0]*255))
        if render:
            env.render() # Cant't use in colab.

        # Store transition and learn.
        total_reward += reward
        print('\rStep: {:3d} | Reward: {:.3f} / {:.3f}'\
            .format(step, reward, total_reward), end="")
            
        state = state_next.copy()
        step += 1
        if done or step>2000:
            print()
            break

    return img_buffer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', '-t', nargs='?', type=str, default="train", help='train / test')
    train_test = parser.parse_args().type
    
    stack_frames = 4
    img_size = (84,84)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_name = "PongNoFrameskip-v4"
    env_ = gym.make(env_name)
    env = PongEnvWrapper(env_, k=stack_frames, img_size=img_size)

    agent = dqn.DeepQNetwork(
        n_actions = env.action_space.n,
        input_shape = [stack_frames, *img_size],
        qnet = models.QNet,
        device = device,
        learning_rate = 2e-4, 
        reward_decay = 0.99,
        replace_target_iter = 1000, 
        memory_size = 10000,
        batch_size = 32,)

    if train_test == "train":
        train(env, agent, stack_frames, img_size, "save", max_steps=400000)
    elif train_test == "test":
        agent.save_load_model(op="load", path="save", fname="qnet.pt")
        img_buffer = play(env, agent, stack_frames, img_size)
        save_gif(img_buffer, "test.gif")
    else:
        print("Wrong args.")