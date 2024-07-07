""" SAC algorithm
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

from stable_baselines3 import SAC
from collections import OrderedDict
from stable_baselines3.common.callbacks import BaseCallback
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
    parser.add_argument('--train-timesteps', default = 250000, type = int, 
                        help = 'number of training timesteps')
    parser.add_argument('--test-episodes', default = 50, type = int, 
                        help = 'number of testing episodes')
    parser.add_argument('--eval-frequency', default = 250, type = int, 
                        help = 'evaluation frequency over training iterations')
    parser.add_argument('--learning-rate', default = 7e-4, type = float, 
                        help = 'learning rate')
    parser.add_argument('--input-model', default = None, type = str, 
                        help = 'pre-trained input model (in .mdl format)')
    parser.add_argument('--directory', default = 'results', type = str, 
                        help = 'path to the output location for checkpoint storage (model and rendering)')
    return parser.parse_args()


class Callback(BaseCallback):

    def __init__(self, agent, env, args, verbose = 1):
        super(Callback, self).__init__(verbose)
        """ initializes a callback object to access the internal state 
        of the reinforcement learning agent over training iterations 
        for performance monitoring

        args:
            agent: reinforcement learning agent
            env: testing environment for performance evaluation
            args: argument parser

        evaluation metrics:
            episode rewards
            episode lengths
        """
        self.test_episodes = args.test_episodes
        self.episode_rewards = list()
        self.episode_lengths = list()
        self.agent = agent
        self.env = env
    
    def _on_step(self) -> bool:
        if self.num_timesteps % args.eval_frequency == 0: 
            episode_rewards, episode_lengths = evaluate_policy(self.agent, 
                                                               self.env, self.test_episodes, 
                                                               return_episode_rewards = True)
            er, el = np.array(episode_rewards), np.array(episode_lengths)
            self.episode_rewards.append(er.mean())
            self.episode_lengths.append(el.mean())
            if self.verbose > 0 and self.num_timesteps % int(args.train_timesteps * 0.25) == 0:
                print(f'training step: {self.num_timesteps} | test episodes: {self.test_episodes} | reward: {er.mean():.2f} +/- {er.std():.2f}')
        return True


def display(frame, step, episode, reward):
    """ renders the (state -> action) agent step in the testing environment 
    as a graphics interchange format frame
    """
    image = Image.fromarray(frame)
    drawer = ImageDraw.Draw(image)
    color = (255, 255, 255) if np.mean(image) < 128 else (0, 0, 0)
    drawer.text((image.size[0] / 20, image.size[1] / 18), 
                f'episode: {episode} | step: {step} | reward: {reward:.2f}', fill = color)
    return image


def multiprocess(args, train_env, test_env, seeds = [1, 2, 3, 5, 8]):
    """ processes multiple sequential training and testing sessions 
    with different seeds (to counteract variance)

    args:
        train_env: training environment
        test_env: testing environment
        seeds: fibonacci seeds
    """
    model = None
    if args.input_model is not None:
        model = args.input_model

    print(f'\nmodel to train: {model}\n')

    pool = {'rewards': list(), 'lengths': list(), 'times': list(), 'weights': list()}
    for iter, seed in enumerate(seeds):
        print(f'\ntraining session: {iter + 1}')
        print("----------------")
        for key, value in zip(pool.keys(), train(args, seed, train_env, test_env, model)):
            pool[key].append(value)
    
    return pool


def train(args, seed, train_env, test_env, model):
    """ trains the agent in the training environment

    args:
        seed: seed of the training session
        model: model to train
    """
    env = gym.make(train_env)
    
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    policy = 'MlpPolicy'

    if model is not None:
        agent = SAC.load(model, 
                         env = env, 
                         seed = seed, 
                         device = args.device)
    else:
        agent = SAC(policy, 
                    env = env, 
                    seed = seed,
                    device = args.device, 
                    learning_rate = args.learning_rate,
                    batch_size = 256, 
                    gamma = 0.99)

    test_env = gym.make(test_env)
    test_env.seed(seed)
    
    callback = Callback(agent, test_env, args)

    start_time = time.time()
    agent.learn(total_timesteps = args.train_timesteps, callback = callback)
    train_time = time.time() - start_time
    
    return callback.episode_rewards, callback.episode_lengths, train_time, agent.policy.state_dict()


def stack(args, metric, records):
    """ stacks training sessions outputs 
    
    args:
        metric: evaluation metric
        records: set of evaluation records
    """
    stacks = [(statistics.mean(elements), statistics.stdev(elements)) 
              for elements in list(zip(*records))]
    xs = np.array([(index + 1) * args.eval_frequency for index in range(len(stacks))])
    ys = np.array([stack[0] for stack in stacks])
    sigmas = np.array([stack[1] for stack in stacks])
    
    return metric, xs, ys, sigmas


def track(metric, xs, ys, sigmas, args):
    """ plots the agent's performance in the testing environment 
    (according to the evaluation metric) over training time-steps
    
    args:
        metric: evaluation metric
        xs: training time-steps (x-axis values)
        ys: set of evaluation records (y-axis values)
        sigmas: set of evaluation records variances
    """
    colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', 
              '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F', '#BAB0AC']
    plt.rcParams['axes.prop_cycle'] = cycler(color = colors)

    plt.plot(xs, ys, alpha = 1, label = f'SAC')
    plt.fill_between(xs, ys - sigmas, ys + sigmas, alpha = 0.5)
  
    plt.xlabel('time-steps')
    plt.ylabel(f'episode {metric}')
    plt.title(f'average episode {metric} over training iterations', loc = 'left')
    plt.grid(True)
    plt.legend()
  
    plt.savefig(f'{args.directory}/SAC-({args.train_env} to {args.test_env})-{metric}.png', dpi = 300)
    plt.close()


def test(args, test_env):
    """ tests the agent in the testing environment """
    env = gym.make(test_env)
    policy = 'MlpPolicy'
    model = None

    if args.train:
        model = f'{args.directory}/SAC-({args.train_env} to {args.test_env}).mdl'
        agent = SAC.load(model, 
                         env = env, 
                         device = args.device)
    else:
        if args.input_model is not None:
            model = args.input_model
            agent = SAC.load(model, 
                             env = env, 
                             device = args.device)
    
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
                frames.append(display(frame, 
                                      step = steps, 
                                      reward = rewards,
                                      episode = num_episodes + 1))
        num_episodes += 1   
        episode_rewards.append(rewards)
    er = np.array(episode_rewards)
    print(f'\ntest episodes: {num_episodes} | reward: {er.mean():.2f} +/- {er.std():.2f}\n')

    if args.render:
        imageio.mimwrite(f'{args.directory}/SAC-({args.train_env} to {args.test_env})-test.gif', frames, fps = 30)

    env.close()


def arrange(args, stacks, train_env):
    """ arranges policy network weights
        
    args:
        stacks: stacks of network weights
    """
    env = gym.make(train_env)
    weights = OrderedDict()
    for key in stacks[0].keys():
        weights[key] = torch.zeros_like(stacks[0][key])
    for weight in stacks:
        for key in weight.keys():
            weights[key] += weight[key]    
    for key in weights.keys():
        weights[key] /= len(weights)
            
    policy = 'MlpPolicy'
    agent = SAC(policy, 
                env = env, 
                device = args.device, 
                learning_rate = args.learning_rate,
                batch_size = 256, 
                gamma = 0.99)
        
    agent.policy.load_state_dict(weights)
    agent.save(f'{args.directory}/SAC-({args.train_env} to {args.test_env}).mdl')
    print(f'\nmodel checkpoint storage: {args.directory}/SAC-({args.train_env} to {args.test_env}).mdl\n')


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
            metric, xs, ys, sigmas = stack(args, metric, records)
            track(metric, xs, ys, sigmas, args)
        print(f'\ntraining time: {np.mean(pool["times"]):.2f} +/- {np.std(pool["times"]):.2f}')
        print("-------------")

        arrange(args, pool['weights'], train_env)
        
    if args.test:
        test(args, test_env)


if __name__ == '__main__':
    main()
