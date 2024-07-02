""" REINFORCE algorithm - Reinforcement Learning (RL) paradigm within the Custom Hopper MuJoCo environment """

import os
import gym
import time
import torch
import imageio
import argparse
import warnings
import statistics
import numpy as np
import matplotlib.pyplot as plt
import PIL.ImageDraw as ImageDraw

import stable_baselines3

from PIL import Image
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
                        default = 25000, 
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


X = 5  # evaluation frequency over training iterations
Y = 25  # verbose output frequency over training iterations


# callback class to evaluate episode rewards and episode lengths over training iterations
class Callback():
    
    def __init__(self, agent, env, args):
        
        self.test_episodes = args.test_episodes
        self.episode_rewards = list()
        self.episode_lengths = list()
        self.agent = agent
        self.env = env
    
    def _on_step(self, num_episodes, verbose = 1) -> bool:
        if num_episodes % X == 0: 
            episode_rewards, episode_lengths = evaluate_policy(self.agent, self.env, self.test_episodes, 
                                                               return_episode_rewards = True)
            er, el = np.array(episode_rewards), np.array(episode_lengths)
            self.episode_rewards.append(er.mean())
            self.episode_lengths.append(el.mean())
            if verbose > 0 and num_episodes % Y == 0:
                print(f'Training Episode: {num_episodes} | Test Episodes: {self.test_episodes} | Avg. Reward: {er.mean():.2f} +/- {er.std():.2f}')
                    
        return True
        

# function to render the simulator
def rendering(frame, steps, num_episodes, rewards):
    image = Image.fromarray(frame)
    drawer = ImageDraw.Draw(image)
    color = (255, 255, 255) if np.mean(image) < 128 else (0, 0, 0)
    drawer.text((image.size[0] / 20, image.size[1] / 18), 
                f'Test Episode: {num_episodes} | Step: {steps} | Reward: {rewards:.2f}', fill = color)
    return image


# function to multiprocess training sessions (to counteract variance)
def pool(args, train_env, test_env, sessions = 8):
    
    env = gym.make(train_env)
    print("---------------------------------------------")
    print(f'Training Environment: {train_env}')
    print("---------------------------------------------")
    print('Action Space:', env.action_space)
    print('State Space:', env.observation_space)
    print('Dynamics Parameters:', env.get_parameters())
    print("---------------------------------------------")
    
    env = gym.make(test_env)
    print("---------------------------------------------")
    print(f'Testing Environment: {test_env}')
    print("---------------------------------------------")
    print('Action Space:', env.action_space)
    print('State Space:', env.observation_space)
    print('Dynamics Parameters:', env.get_parameters())
    print("---------------------------------------------")
  
    model = None
    if args.input_model is not None:
        model = args.input_model
      
    print("---------------------------------------------")
    print('Model to Train:', model)
    print("---------------------------------------------")

    pool = {'rewards': list(), 'lengths': list(), 'times': list(), 'weights': list()}
    
    for iter in range(sessions):
        print("---------------------------------------------")
        print('Training Session:', iter)
        print("---------------------------------------------")
        
        for key, value in zip(pool.keys(), [train(args, train_env, test_env, model)]):
            pool[key].append(value)
    
    return pool


# function to train the simulator
def train(args, train_env, test_env, model):
  
    """ Training """
  
    env = gym.make(train_env)
    policy = RFPolicy(env.observation_space.shape[-1], env.action_space.shape[-1])

    if model is not None:
        policy.load_state_dict(torch.load(model), 
                               strict = True)
    agent = RF(policy, 
               device = args.device, 
               baseline = args.baseline)

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
    
    return callback.episode_rewards, callback.episode_lengths, time.time() - start, policy.state_dict()
        

def aggregate(metric, records):
    averages = [(statistics.mean(elements), statistics.stdev(elements)) for elements in list(zip(*records))]
    xs = np.insert(np.array([index * X for index in range(len(averages))]), 0, 0)
    ys = np.insert(np.array([element[0] for element in averages]), 0, 0)
    sigmas = np.insert(np.array([element[1] for element in averages]), 0, 0)
    return metric, sx, ys, sigmas


def plot(metric, xs, ys, sigmas, args):
  
    plt.plot(xs, ys, alpha = 1, label = f'RF {args.baseline}')
    plt.fill_between(xs, ys - sigmas, ys + sigmas, alpha = 0.5)
  
    plt.xlabel('episodes')
    plt.ylabel(f'episode {metric}')
    plt.title(f'average episode {metric} over training iterations', loc = 'left')
    plt.legend()
  
    plt.savefig(f'{args.directory}/RF-{args.baseline}-({args.train_env} to {args.test_env})-{metric}.png', dpi = 300)
    plt.close()


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
    print('Model to Test:', model)
    print("---------------------------------------------")

    frames = list()
    num_episodes = 0
    episode_rewards = []
    while num_episodes < args.test_episodes:
        done = False
        obs = env.reset()
        rewards, steps = (0, 0)
        while not done:
            action, _ = agent.predict(obs, deterministic = True)
            next_state, reward, done, _ = env.step(action)
            rewards += reward
            obs = next_state
            
            steps += 1
            if args.render and num_episodes < 5:
                frame = env.render(mode = 'rgb_array')
                frames.append(rendering(frame, steps, num_episodes + 1, rewards))
                
        num_episodes += 1   
        episode_rewards.append(rewards)
    er = np.array(episode_rewards)
    print("---------------------------------------------")
    print(f'Test Episodes: {num_episodes} | Avg. Reward: {er.mean():.2f} +/- {er.std():.2f}')
    print("---------------------------------------------")

    if args.render:
        imageio.mimwrite(f'{args.directory}/RF-{args.baseline}-({args.train_env} to {args.test_env})-test.gif', frames, fps = 30)

    env.close()


def main():
    args = parse_args()
    warnings.filterwarnings("ignore")
    
    if not os.path.exists(args.directory):
        os.mkdir(args.directory)

    train_env, test_env = tuple(f'CustomHopper-{x}-v0' 
                                for x in [args.train_env, 
                                          args.test_env])

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("---------------------------------------------")
        print('WARNING: GPU not available, switch to CPU')
        args.device = 'cpu'
        
    # validate environment registration
    try: env = gym.make(train_env)
    except gym.error.UnregisteredEnv: 
        raise ValueError(f'ERROR: environment {train_env} not found')
        
    try: env = gym.make(test_env)
    except gym.error.UnregisteredEnv: 
        raise ValueError(f'ERROR: environment {test_env} not found')
        
    # validate model loading
    if args.input_model is not None and not os.path.isfile(args.input_model):
        raise FileNotFoundError(f'ERROR: model file {args.input_model} not found')
      
    if args.train:
        pool = pool(args, train_env, test_env)
        for metric, records in zip(('reward', 'length'), (pool['rewards'], pool['lengths'])):
            metric, xs, ys, sigmas = aggregate(metric, records)
            plot(metric, xs, ys, sigmas, args)
        print("---------------------------------------------")
        print(f'Training Time: {np.mean(pool["times"]):.2f} +/- {np.std(pool["times"]):.2f}')
        print("---------------------------------------------")
        
        weights = {}
        for key in pool['weights'][0].keys():
            weights[key] = torch.mean(torch.stack([w[key] for w in weights]), dim = 0)
            
        policy = RFPolicy(env.observation_space.shape[-1], env.action_space.shape[-1])
        policy.load_state_dict(weights)
        torch.save(policy.state_dict(), f'{args.directory}/RF-{args.baseline}-({args.train_env} to {args.test_env}).mdl')
        
    if args.test:
        test(args, test_env)


if __name__ == '__main__':
    main()
