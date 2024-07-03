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
    parser.add_argument('--train-env', default = 'source', type = str, 
			help = 'Training environment')
    parser.add_argument('--test-env', default = 'target', type = str, 
			help = 'Testing environment')
    parser.add_argument('--train-timesteps', default = 100000, type = int, 
			help = 'Number of training timesteps')
    parser.add_argument('--test-episodes', default = 100, type = int, 
			help = 'Number of testing episodes')
    parser.add_argument('--device', default = 'cpu', type = str, 
			help = 'Network device [cpu, cuda]')
    return parser.parse_args()


def train(device = 'cpu', 
          train_timesteps: int = 100000, 
          train_env: str = 'CustomHopper-source-v0', **kwargs) -> SAC:
    """ 
        -> train the agent in the training environment

    """
    env = gym.make(train_env)
    policy = 'MlpPolicy'
    agent = SAC(policy = policy, env = env, device = device, **kwargs)
    
    agent.learn(total_timesteps = train_timesteps)
      
    return agent


def test(agent: SAC, 
         test_episodes: int = 100, 
         test_env: str = 'CustomHopper-source-v0') -> float:
    """ 
        -> test the agent in the testing environment
        
    """  
    env = gym.make(test_env)
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
                test_env = test_env), kwargs



def gridsearch(args, params, train_env, test_env, sessions = 5):
    results = []
    keys = list(params.keys())
    for param in list(itertools.product(*params.values())):
        kwargs = dict(zip(keys, param))
        pool = list()
        for iter in range(sessions):
            er, _ = pooling(kwargs, 
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
