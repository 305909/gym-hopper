""" REINFORCE algorithm
Custom Hopper MuJoCo environment
"""

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
from cycler import cycler
from env.custom_hopper import *

from collections import OrderedDict
from agents.rein import RF, RFPolicy
from stable_baselines3.common.evaluation import evaluate_policy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action = 'store_true', 
                        help = 'train the model')
    parser.add_argument('--test', action = 'store_true', 
                        help = 'test the model')
    parser.add_argument('--render', action = 'store_true', 
                        help = 'render the simulator')
    parser.add_argument('--device', default = 'cpu', type = str, 
                        help = 'network device [cpu, cuda]')
    parser.add_argument('--train-env', default = 'source', type = str, 
                        help = 'training environment')
    parser.add_argument('--test-env', default = 'target', type = str, 
                        help = 'testing environment')
    parser.add_argument('--train-episodes', default = 25000, type = int, 
                        help = 'number of training episodes')
    parser.add_argument('--test-episodes', default = 50, type = int, 
                        help = 'number of testing episodes')
    parser.add_argument('--baseline', default = 'vanilla', type = str, 
                        choices = ['vanilla', 'constant', 'whitening'], 
                        help = 'baseline for the policy update function [vanilla, constant, whitening]')
    parser.add_argument('--input-model', default = None, type = str, 
                        help = 'pre-trained input model (in .mdl format)')
    parser.add_argument('--directory', default = 'results', type = str, 
                        help = 'path to the output location for checkpoint storage (model and rendering)')
    return parser.parse_args()


X = 10 # 250
Y = 50 # 6250


class Callback():
    """ 
        -> evaluate the agent over training iterations
        -> (testing environment) 
        
    ----------
    X: evaluation frequency over training iterations
    Y: verbose output frequency over training iterations
    """
    def __init__(self, agent, env, args):
        self.test_episodes = args.test_episodes
        self.episode_rewards = list()
        self.episode_lengths = list()
        self.agent = agent
        self.env = env
    
    def _on_step(self, num_episodes, verbose = 1) -> bool:
        """ 
            -> evaluate the agent after X training episodes
            -> output evaluation information after Y episodes

        ----------
        evaluation metrics: episode rewards, episode lengths
        """
        if num_episodes % X == 0: 
            episode_rewards, episode_lengths = evaluate_policy(self.agent, self.env, self.test_episodes, 
                                                               return_episode_rewards = True)
            er, el = np.array(episode_rewards), np.array(episode_lengths)
            self.episode_rewards.append(er.mean())
            self.episode_lengths.append(el.mean())
            if verbose > 0 and num_episodes % Y == 0:
                print(f'training episode: {num_episodes} | test episodes: {self.test_episodes} | reward: {er.mean():.2f} +/- {er.std():.2f}')
        return True
        

def display(frame, steps, num_episodes, rewards):
    """ 
        -> display the (state-action) agent step
        -> (testing environment)
        -> output a graphic interchange format frame

    ----------
    image: graphic interchange format frame
    """
    image = Image.fromarray(frame)
    drawer = ImageDraw.Draw(image)
    color = (255, 255, 255) if np.mean(image) < 128 else (0, 0, 0)
    drawer.text((image.size[0] / 20, image.size[1] / 18), 
                f'episode: {num_episodes} | step: {steps} | reward: {rewards:.2f}', fill = color)
    return image


def multiprocess(args, train_env, test_env, sessions = 5):
    """ 
        -> multiprocess sequential training sessions
        -> (counteract variance)

    ----------
    sessions: sequential training sessions to process
    pool: dictionary of training session outputs
    """
    model = None
    if args.input_model is not None:
        model = args.input_model

    print(f'\nmodel to train: {model}\n')

    pool = {'rewards': list(), 'lengths': list(), 'times': list(), 'weights': list()}
    for iter in range(sessions):
        print(f'\ntraining session: {iter + 1}')
        print("----------------")
        for key, value in zip(pool.keys(), train(args, train_env, test_env, model)):
            pool[key].append(value)
    
    return pool


def train(args, train_env, test_env, model):
    """ 
        -> train the agent
        -> (training environment)

    ----------
    model: model to train
    """
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
        

def stack(metric, records):
    """ 
        -> stak training sessions outputs 

    """
    stacks = [(statistics.mean(elements), statistics.stdev(elements)) 
              for elements in list(zip(*records))]
    xs = np.array([(index + 1) * X for index in range(len(stacks))])
    ys = np.array([stack[0] for stack in stacks])
    sigmas = np.array([stack[1] for stack in stacks])
    
    return metric, xs, ys, sigmas


def track(metric, xs, ys, sigmas, args):
    """ 
        -> plot the evaluation metric progress over the training episodes

    """
    colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', 
              '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F', '#BAB0AC']
    plt.rcParams['axes.prop_cycle'] = cycler(color = colors)

    plt.plot(xs, ys, alpha = 1, label = f'RF {args.baseline}')
    plt.fill_between(xs, ys - sigmas, ys + sigmas, alpha = 0.5)
  
    plt.xlabel('episodes')
    plt.ylabel(f'episode {metric}')
    plt.title(f'average episode {metric} over training iterations', loc = 'left')
    plt.legend()
  
    plt.savefig(f'{args.directory}/RF-{args.baseline}-({args.train_env} to {args.test_env})-{metric}.png', dpi = 300)
    plt.close()


def test(args, test_env):
    """ 
        -> test the agent
        -> (testing environment)
        
    """
    env = gym.make(test_env)
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

    print(f'\nmodel to test: {model}\n')

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
                frames.append(display(frame, steps, num_episodes + 1, rewards))
                
        num_episodes += 1   
        episode_rewards.append(rewards)
    er = np.array(episode_rewards)
    print(f'\ntest episodes: {num_episodes} | reward: {er.mean():.2f} +/- {er.std():.2f}\n')

    if args.render:
        imageio.mimwrite(f'{args.directory}/RF-{args.baseline}-({args.train_env} to {args.test_env})-test.gif', frames, fps = 30)

    env.close()


def arrange(args, stacks, train_env):
    """ 
        -> arrange policy network weights
        -> save the model
        
    ----------
    stacks: stacks of network weights
    """
    env = gym.make(train_env)
    weights = OrderedDict()
    for key in stacks[0].keys():
        weights[key] = torch.mean(torch.stack([w[key] 
                                               for w in stacks]), dim = 0)

    policy = RFPolicy(env.observation_space.shape[-1], env.action_space.shape[-1])
    policy.load_state_dict(weights)
        
    torch.save(policy.state_dict(), f'{args.directory}/RF-{args.baseline}-({args.train_env} to {args.test_env}).mdl')
    print(f'\nmodel checkpoint storage: {args.directory}/RF-{args.baseline}-({args.train_env} to {args.test_env}).mdl\n')


def main():
    args = parse_args()
    warnings.filterwarnings("ignore")
    
    if not os.path.exists(args.directory):
        os.mkdir(args.directory)

    train_env, test_env = tuple(f'CustomHopper-{x}-v0' 
                                for x in [args.train_env, 
                                          args.test_env])

    if args.device == 'cuda' and not torch.cuda.is_available():
        print('\nWARNING: GPU not available, switch to CPU\n')
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
        pool = multiprocess(args, train_env, test_env)
        for metric, records in zip(('reward', 'length'), (pool['rewards'], pool['lengths'])):
            metric, xs, ys, sigmas = stack(metric, records)
            track(metric, xs, ys, sigmas, args)
        print(f'\ntraining time: {np.mean(pool["times"]):.2f} +/- {np.std(pool["times"]):.2f}')
        print("-------------")
        
        arrange(args, pool['weights'], train_env)
        
    if args.test:
        test(args, test_env)


if __name__ == '__main__':
    main()
