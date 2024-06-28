""" REINFORCE algorithm - Reinforcement Learning (RL) paradigm within the Custom Hopper MuJoCo environment """

import os
import gym
import time
import torch
import imageio
import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt
import PIL.ImageDraw as ImageDraw

import stable_baselines3

from PIL import Image
from cycler import cycler
from env.custom_hopper import *

from agents.rein import RF, RFPolicy
from stable_baselines3.common.evaluation import evaluate_policy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', 
                        action = 'store_true', 
                        help = 'Train the model')
    parser.add_argument('--test', 
                        action = 'store_true', 
                        help = 'Test the model')
    parser.add_argument('--render', 
                        action = 'store_true', 
                        help = 'Render the simulator')
    parser.add_argument('--device', 
                        default = 'cpu', 
                        type = str, 
                        choices = ['cpu', 
                                   'cuda'], 
                        help = 'Network device [cpu, cuda]')
    parser.add_argument('--train-env', 
                        default = 'source', 
                        type = str,
                        choices = ['source', 
                                   'target'],
                        help = 'Training environment')
    parser.add_argument('--test-env', 
                        default = 'target', 
                        type = str,
                        choices = ['source', 
                                   'target'],
                        help = 'Testing environment')
    parser.add_argument('--train-episodes', 
                        default = 100000, 
                        type = int, 
                        help = 'Number of training episodes')
    parser.add_argument('--test-episodes', 
                        default = 50, 
                        type = int, 
                        help = 'Number of testing episodes')
    parser.add_argument('--baseline', 
                        default = 'vanilla', 
                        type = str, 
                        choices = ['vanilla', 
                                   'constant', 
                                   'whitening'], 
                        help = 'Baseline for the policy update function [vanilla, constant, whitening]')
    parser.add_argument('--input-model', 
                        default = None, 
                        type = str, 
                        help = 'Pre-trained input model (in .mdl format)')
    parser.add_argument('--directory', 
                        default = 'results', 
                        type = str, 
                        help = 'Path to the output location for checkpoint storage (model and rendering)')
    return parser.parse_args()


# callback class to evaluate rewards over training iterations
class Callback():
    
    def __init__(self, agent, env, args, verbose = 1):
        
        self.train_episodes = args.train_episodes
        self.test_episodes = args.test_episodes
        self.episode_rewards = list()
        self.episode_lengths = list()
        self.verbose = verbose
        self.agent = agent
        self.args = args
        self.env = env
        
    def _on_step(self, num_episodes) -> bool:
        if num_episodes % (self.train_episodes * 1e-3) == 0: 
            episode_rewards, episode_lengths = evaluate_policy(self.agent, self.env, self.test_episodes, 
                                                               return_episode_rewards = True)
            
            er, el = np.array(episode_rewards), np.array(episode_lengths)
            self.episode_rewards.append((er.mean(), 
                                         er.mean() - er.std(), 
                                         er.mean() + er.std()))
            self.episode_lengths.append((el.mean(), 
                                         el.mean() - el.std(), 
                                         el.mean() + el.std()))
            if self.verbose > 0 and num_episodes % (self.train_episodes * 1e-2) == 0:
                print(f'Training Episodes: {num_episodes - int(self.train_episodes * 1e-2)} - {num_episodes} | Test Episodes: {self.test_episodes} | Avg. Reward: {er.mean():.2f} +/- {er.std():.2f}')
        return True
        

# function to render the simulator
def rendering(frame, steps, episode, rewards):
    image = Image.fromarray(frame)
    drawer = ImageDraw.Draw(image)
    color = (255, 255, 255) if np.mean(image) < 128 else (0, 0, 0)
    drawer.text((image.size[0] / 20, image.size[1] / 18), 
                f'Episode: {episode + 1} | Step: {steps + 1} | Reward: {rewards:.2f}', fill = color)
    return image


# function to train the simulator
def train(args, train_env, test_env):
    env = gym.make(train_env)

    print("---------------------------------------------")
    print(f'Training Environment: {train_env}')
    print("---------------------------------------------")
    print('Action Space:', env.action_space)
    print('State Space:', env.observation_space)
    print('Dynamics Parameters:', env.get_parameters())
    print("---------------------------------------------")

    """ Training """
    
    policy = RFPolicy(env.observation_space.shape[-1], env.action_space.shape[-1])
    model = None

    if args.input_model is not None:
        model = args.input_model
        policy.load_state_dict(torch.load(model), 
                               strict = True)
    agent = RF(policy, 
               device = args.device, 
               baseline = args.baseline)

    print("---------------------------------------------")
    print('Model to train:', model)
    print("---------------------------------------------")

    callback = Callback(agent, gym.make(test_env), args)
    num_episodes = 0
    
    start = time.time()
    while num_episodes < args.train_episodes:
        done = False
        obs = env.reset()
        while not done:
            action, action_log_prob = agent.predict(obs)
            next_state, reward, done, _ = env.step(action)
            agent.store_outcome(obs, action_log_prob, reward)
            obs = next_state 
        num_episodes += 1   
        agent.update_policy()
        callback._on_step(num_episodes)

    torch.save(agent.policy.state_dict(), f'{args.directory}/RF-{args.baseline}-({args.train_env} to {args.test_env}).mdl')
    print("---------------------------------------------")
    print(f'Time: {time.time() - start:.2f}')
    print("---------------------------------------------")
    
    # exponential moving average
    def smooth(scalars, weight = 0.85):
        x = list()
        last = scalars[0]
        for point in scalars:
            value = last * weight + (1 - weight) * point
            x.append(value)
            last = value
        return x

    for metric, records in zip(('reward', 'length'), (callback.episode_rewards, callback.episode_lengths)):
        x, y = [0], [0]
        uppers, lowers = [0], [0]
        
        for key, value in enumerate(records):
            point = key * int(self.train_episodes * 1e-3)
            x.append(point)
            y.append(value[0])
            lowers.append(value[1])
            uppers.append(value[2])

        colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', 
                  '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F', '#BAB0AC']
        plt.rcParams['axes.prop_cycle'] = cycler(color = colors)
    
        plt.plot(x, smooth(y), alpha = 1, label = f'RF {args.baseline}')
        plt.fill_between(x, smooth(lowers), smooth(uppers), alpha = 0.5)
        
        plt.xlabel('episodes')
        plt.ylabel(f'episode {metric}')
        plt.title(f'average episode {metric} over training iterations', loc = 'left')
        plt.legend()
        
        plt.savefig(f'{args.directory}/RF-{args.baseline}-({args.train_env} to {args.test_env})-{metric}.png', dpi = 300)
        plt.close()
    
    env.close()


# function to test the simulator
def test(args, test_env):
    env = gym.make(test_env)

    print("---------------------------------------------")
    print(f'Testing Environment: {test_env}')
    print("---------------------------------------------")
    print('Action Space:', env.action_space)
    print('State Space:', env.observation_space)
    print('Dynamics Parameters:', env.get_parameters())
    print("---------------------------------------------")

    """ Evaluation """
    
    policy = RFPolicy(env.observation_space.shape[-1], env.action_space.shape[-1])
    model = None

    if args.train:
        model = f'{args.directory}/RF-{args.baseline}-({args.train_env} to {args.test_env}).mdl'
        policy.load_state_dict(torch.load(model), 
                               strict = True)
    else:
        if args.input_model is not None:
            model = args.input_model
            policy.load_state_dict(torch.load(model), 
                               strict = True)
    agent = RF(policy, 
               device = args.device, 
               baseline = args.baseline)
    
    print("---------------------------------------------")
    print('Model to test:', model)
    print("---------------------------------------------")

    frames = []
    episode_rewards = []
    for episode in range(args.test_episodes):
        done = False
        obs = env.reset()
        rewards, steps = (0, 0)
        while not done:
            action, _ = agent.predict(obs, deterministic = True)
            next_state, reward, done, _ = env.step(action)
            rewards += reward
            obs = next_state

            if args.render and episode < 10:
                frame = env.render(mode = 'rgb_array')
                frames.append(rendering(frame, steps, episode, rewards))
                steps += 1

        episode_rewards.append(rewards)
    er = np.array(episode_rewards)
    print("---------------------------------------------")
    print(f'Test Episodes: {episode + 1} | Avg. Reward: {er.mean():.2f} +/- {er.std():.2f}')
    print("---------------------------------------------")

    if args.render:
        imageio.mimwrite(f'{args.directory}/RF-{args.baseline}-({args.train_env} to {args.test_env}).gif', frames, fps = 30)

    env.close()


def main():
    args = parse_args()
    warnings.filterwarnings("ignore")

    if not os.path.exists(args.directory):
        os.mkdir(args.directory)

    train_env, test_env = tuple(f'CustomHopper-{x}-v0' for x in [args.train_env, args.test_env])

    if args.train:
        train(args, train_env, test_env)

    if args.test or args.render:
        test(args, test_env)


if __name__ == '__main__':
    main()
