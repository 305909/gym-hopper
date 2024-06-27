import os
import sys
import gym
import argparse
import warnings
import itertools
import numpy as np

sys.path.append(
	os.path.abspath(
		os.path.join(os.path.dirname(__file__), '..')))

import stable_baselines3

from stable_baselines3 import SAC
from env.custom_hopper import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-env', 
                        default = 'source', type = str,
                        choices = ['source', 
                                   'target',
                                   'source-moderate-randomization', 
                                   'target-moderate-randomization',
                                   'source-wide-randomization', 
                                   'target-wide-randomization'],
                        help = 'Training environment')
    parser.add_argument('--test-env', 
                        default = 'target', type = str,
                        choices = ['source', 
                                   'target',
                                   'source-moderate-randomization', 
                                   'target-moderate-randomization',
                                   'source-wide-randomization', 
                                   'target-wide-randomization'],
                        help = 'Testing environment')
    parser.add_argument('--train-timesteps', 
                        default = 19750, 
                        type = int, 
                        help = 'Number of training episodes')
    parser.add_argument('--test-episodes', 
                        default = 250, 
                        type = int, 
                        help = 'Number of testing episodes')
    parser.add_argument('--device', 
                        default = 'cpu', 
                        type = str, 
                        choices = ['cpu', 
                                   'cuda'], 
                        help = 'Network device [cpu, cuda]')
    return parser.parse_args()


def train(device = 'cpu', 
          train_timesteps: int = 19750, 
          train_env: str = 'CustomHopper-source-v0', **kwargs) -> SAC:
          
    env = gym.make(train_env)

    """ Training """
              
    policy = 'MlpPolicy'
    agent = SAC(policy = policy, env = env, device = device, **kwargs, verbose = False)
    
    agent.learn(total_timesteps = train_timesteps)
      
    return agent


def test(agent: SAC, 
         test_episodes: int = 250, 
         test_env: str = 'CustomHopper-source-v0') -> float:
             
    env = gym.make(test_env)
  
    """ Evaluation """

    num_episodes = 0
    episode_rewards = []
    while num_episodes < test_episodes:
        done = False
        obs = env.reset()
        rewards, steps = (0, 0)
        while not done:
            action, _ = agent.predict(obs, deterministic = True)
            next_state, reward, done, _ = env.step(action)
            rewards += reward
            obs = next_state
        num_episodes += 1
        episode_rewards.append(rewards)
    er = np.array(episode_rewards)
    
    return er


def pooling(kwargs: dict, device, train_timesteps, test_episodes, train_env, test_env):
    
    agent = train(device = device, 
                  train_timesteps = train_timesteps, 
                  train_env = train_env, **kwargs)
    
    return test(agent, 
                test_episodes = test_episodes, 
                test_env = test_env)



def gridsearch(args, params, train_env, test_env):
    results = []
    keys = list(params.keys())
    for param in itertools.product(*params.values()):
        kwargs = dict(zip(keys, param))
        er = pooling(kwargs, 
                     device = args.device,
                     train_timesteps = args.train_timesteps,
                     test_episodes = args.test_episodes, 
                     train_env = train_env, 
                     test_env = test_env)
        cov = er.std() / er.mean()  # coefficient of variation
        score = er.mean() * (1 - cov)
        print("---------------------------------------------")
        print(f'Score: {score:.2f} | Avg. Reward: {er.mean():.2f} +/- {er.std():.2f} | Parameters: {kwargs}')
        results.append([score, er.mean(), er.std(), kwargs])

    results.sort(key = lambda x: x[0], reverse = True)
    print("---------------------------------------------")
    print(f'Grid Search - Ranking Scores:')
    print("---------------------------------------------")
    for rank, candidate in enumerate(results):
        print(f'{rank + 1} | Score: {candidate[0]:.2f} | Avg. Reward: {candidate[1]:.2f} +/- {candidate[2]:.2f} | Parameters: {candidate[3]}')
        print("---------------------------------------------")

    return max(results, key = lambda x: x[0])


def main():
    args = parse_args()
    warnings.filterwarnings("ignore")
    params = {                                           # | source -> source | source -> target | target -> target
        'batch_size': [64, 128, 256],                    # | ...              | ...              | ...
        'learning_rate': [1e-3, 7e-4, 5e-4, 3e-4, 1e-4]  # | ...              | ...              | ...
        }
    
    train_env, test_env = tuple(f'CustomHopper-{x}-v0' for x in [args.train_env, args.test_env])
    
    prime = gridsearch(args, params, train_env, test_env)
    print("---------------------------------------------")
    print(f'Maximum Score: {prime[0]:.2f} | Avg. Reward: {prime[1]:.2f} +/- {prime[2]:.2f} | Optimal Parameters: {prime[3]}')
    print("---------------------------------------------")


if __name__ == '__main__':
    main()
