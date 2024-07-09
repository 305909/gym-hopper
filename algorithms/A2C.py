""" A2C algorithm
Custom Hopper 
MuJoCo environment
"""

import os
import gym
import time
import torch
import utils
import imageio
import argparse
import warnings
import statistics
import numpy as np
import matplotlib.pyplot as plt
import PIL.ImageDraw as ImageDraw

import stable_baselines3

sys.path.append(
	os.path.abspath(
		os.path.join(os.path.dirname(__file__), '..')))

from PIL import Image
from cycler import cycler
from env.custom_hopper import *

from collections import OrderedDict
from agents.aac import A2C, A2CPolicy
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
    parser.add_argument('--test-env', default = 'source', type = str, 
                        help = 'testing environment')
    parser.add_argument('--train-episodes', default = 2500, type = int, 
                        help = 'number of training episodes')
    parser.add_argument('--test-episodes', default = 10, type = int, 
                        help = 'number of testing episodes')
    parser.add_argument('--eval-frequency', default = 10, type = int, 
                        help = 'evaluation frequency over training iterations')
    parser.add_argument('--input-model', default = None, type = str, 
                        help = 'pre-trained input model (in .mdl format)')
    parser.add_argument('--directory', default = 'results', type = str, 
                        help = 'path to the output location for checkpoint storage (model and rendering)')
    return parser.parse_args()


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

    policy = A2CPolicy(env.observation_space.shape[-1], env.action_space.shape[-1])

    if model is not None:
        policy.load_state_dict(torch.load(model), 
                               strict = True)
    agent = A2C(policy, 
                device = args.device)
    
    test_env = gym.make(test_env)
    test_env.seed(seed)
    
    callback = utils.Callback(agent, test_env, args)
    
    num_episodes = 0
    num_timesteps = 0
    callback._on_step(num_episodes, args)
    
    start_time = time.time()
    while num_episodes < args.train_episodes:
        env.seed(seed)
        
        done = False
        obs = env.reset()
        while not done:
            action, action_log_prob, state_value = agent.predict(obs)
            next_state, reward, done, _ = env.step(action)
            agent.store_outcome(obs, action_log_prob, reward, done, state_value)
            obs = next_state
            num_timesteps += 1
            if num_timesteps % agent.batch_size == 0: 
                agent.update_policy()
                
        num_episodes += 1   
        callback._on_step(num_episodes, args)
    train_time = time.time() - start_time
    
    return callback.episode_rewards, callback.episode_lengths, train_time, policy.state_dict()


def test(args, test_env):
    """ tests the agent in the testing environment """
    env = gym.make(test_env)
    policy = A2CPolicy(env.observation_space.shape[-1], env.action_space.shape[-1])
    model = None

    if args.train:
        model = f'{args.directory}/A2C-({args.train_env} to {args.test_env}).mdl'
        policy.load_state_dict(torch.load(model), 
                               strict = True)
    else:
        if args.input_model is not None:
            model = args.input_model
            policy.load_state_dict(torch.load(model), 
                                   strict = True)
    agent = A2C(policy, 
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
                frames.append(utils.display(frame, 
                                            step = steps, 
                                            reward = rewards,
                                            episode = num_episodes + 1))
        num_episodes += 1   
        episode_rewards.append(rewards)
    er = np.array(episode_rewards)
    print(f'\ntest episodes: {num_episodes} | reward: {er.mean():.2f} +/- {er.std():.2f}\n')

    if args.render:
        imageio.mimwrite(f'{args.directory}/A2C-({args.train_env} to {args.test_env})-test.gif', frames, fps = 30)

    env.close()


def arrange(args, stacks, train_env):
    """ arranges policy network weights
        
    args:
        stacks: stacks of network weights
    """
    env = gym.make(train_env)
    weights = OrderedDict()
    for key in stacks[0].keys():
        subkeys = stacks[0][key].keys()
        weights[key] = OrderedDict()
        for subkey in subkeys:
            weights[key][subkey] = torch.mean(torch.stack([w[key][subkey] 
                                                           for w in stacks]), dim = 0)

    policy = A2CPolicy(env.observation_space.shape[-1], env.action_space.shape[-1])
    policy.load_state_dict(weights)
        
    torch.save(policy.state_dict(), f'{args.directory}/A2C-({args.train_env} to {args.test_env}).mdl')
    print(f'\nmodel checkpoint storage: {args.directory}/A2C-({args.train_env} to {args.test_env}).mdl\n')


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
        pool = utils.multiprocess(args, train_env, test_env)
        for metric, records in zip(('reward', 'length'), (pool['rewards'], pool['lengths'])):
            metric, xs, ys, sigmas = utils.stack(args, metric, records)
            if metric == 'reward':
                path = os.path.join(args.directory, f'A2C-({args.train_env} to {args.test_env})-rewards.npy')
                np.save(path, ys)
            utils.track(metric, xs, ys, sigmas, args, f'A2C-({args.train_env} to {args.test_env})-{metric}')
        print(f'\ntraining time: {np.mean(pool["times"]):.2f} +/- {np.std(pool["times"]):.2f}')
        print("-------------")

        arrange(args, pool['weights'], train_env)
        
    if args.test:
        test(args, test_env)


if __name__ == '__main__':
    main()