import os
import sys
import gym
import torch
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
    parser.add_argument('--train-env', default = 'source', type = str, 
			help = 'training environment')
    parser.add_argument('--test-env', default = 'target', type = str, 
			help = 'testing environment')
    parser.add_argument('--train-timesteps', default = 100000, type = int, 
			help = 'number of training timesteps')
    parser.add_argument('--test-episodes', default = 100, type = int, 
			help = 'number of testing episodes')
    parser.add_argument('--device', default = 'cpu', type = str, 
			help = 'network device [cpu, cuda]')
    return parser.parse_args()


def train(seed: int, 
	  device: str = 'cpu', 
          train_timesteps: int = 100000, 
          train_env: str = 'CustomHopper-source-v0', **kwargs) -> SAC:
    """ trains the agent in the training environment """ 
    env = gym.make(train_env)
		  
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
		  
    policy = 'MlpPolicy'
    agent = SAC(policy, 
		env = env, 
		seed = seed, 
		device = device, **kwargs)
    
    agent.learn(total_timesteps = train_timesteps)
      
    return agent


def test(seed: int, agent: SAC, 
         test_episodes: int = 100, 
	 test_env: str = 'CustomHopper-source-v0') -> float:
    """ tests the agent in the testing environment """ 
    env = gym.make(test_env)
		  
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
		 
    num_episodes = 0
    episode_rewards = []
    while num_episodes < test_episodes:
        env.seed(seed)
	    
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


def pooling(kwargs: dict, seed, device, train_timesteps, test_episodes, 
	    train_env, test_env):
    
    agent = train(seed = seed, 
		  device = device, 
                  train_timesteps = train_timesteps, 
                  train_env = train_env, **kwargs)
    
    return test(seed, agent,
                test_episodes = test_episodes, 
                test_env = test_env), kwargs



def gridsearch(args, params, train_env, test_env, seeds = [1, 2, 3, 5, 8]):
    results = []
    keys = list(params.keys())
    for param in list(itertools.product(*params.values())):
        kwargs = dict(zip(keys, param))
        pool = list()
        for iter, seed in enumerate(seeds):
            er, _ = pooling(kwargs, 
			    seed = seed,
			    device = args.device,
                            train_timesteps = args.train_timesteps,
                            test_episodes = args.test_episodes, 
			    train_env = train_env, 
			    test_env = test_env)
            pool.append(er)
        pool = np.array(pool)
        res = np.mean(pool, axis = 0)
        cov = res.std() / res.mean()  # coefficient of variation
        score = res.mean() * (1 - cov)
        print(f'score: {score:.2f} | reward: {res.mean():.2f} +/- {res.std():.2f} | parameters: {kwargs}')
        results.append([score, res.mean(), res.std(), kwargs])

    results.sort(key = lambda x: x[0], reverse = True)
    print(f'\ngrid search - ranking scores:')
    print("----------------------------")
    for rank, candidate in enumerate(results):
        print(f'{rank + 1} | score: {candidate[0]:.2f} | reward: {candidate[1]:.2f} +/- {candidate[2]:.2f} | parameters: {candidate[3]}')

    return max(results, key = lambda x: x[0])


def main():
    args = parse_args()
    warnings.filterwarnings("ignore")
    params = {
        'learning_rate': [1e-3, 7e-4, 5e-4, 3e-4, 1e-4]
        }
    
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
    
    prime = gridsearch(args, params, train_env, test_env)
    print(f'\nmaximum score: {prime[0]:.2f} | reward: {prime[1]:.2f} +/- {prime[2]:.2f} | optimal parameters: {prime[3]}')
    print("-------------")


if __name__ == '__main__':
    main()
