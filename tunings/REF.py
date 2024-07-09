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

from agents.rein import RF, RFPolicy
from env.custom_hopper import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-episodes', default = 10000, type = int, 
                        help = 'number of training episodes')
    parser.add_argument('--test-episodes', default = 100, type = int, 
                        help = 'number of testing episodes')
    parser.add_argument('--device', default = 'cpu', type = str, 
			help = 'network device [cpu, cuda]')
    return parser.parse_args()


def train(seed: int, 
	  device: str = 'cpu', 
          train_episodes: int = 10000, 
          train_env: str = 'CustomHopper-source-v0', **kwargs) -> RF:
    """ trains the agent in the training environment """   
    env = gym.make(train_env)
		  
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
		  
    policy = RFPolicy(env.observation_space.shape[-1], env.action_space.shape[-1], **kwargs)
    agent = RF(policy, 
	       device = device, **kwargs)
              
    num_episodes = 0
    while num_episodes < train_episodes:
        env.seed(seed)
	    
        done = False
        obs = env.reset()
        while not done:
            action, action_log_prob = agent.predict(obs)
            next_state, reward, done, _ = env.step(action)
            agent.store_outcome(obs, action_log_prob, reward)
            obs = next_state 
        num_episodes += 1   
        agent.update_policy()
      
    return agent


def test(seed: int, agent: RF, 
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


def pooling(kwargs: dict, seed, device, train_episodes, test_episodes):
    
    agent = train(seed = seed,
		  device = device, 
                  train_episodes = train_episodes, **kwargs)
    
    return test(seed, agent,
                test_episodes = test_episodes), kwargs


def gridsearch(args, params, seeds = [1, 2, 3]):
    results = []
    keys = list(params.keys())
    for param in list(itertools.product(*params.values())):
        kwargs = dict(zip(keys, param))
        pool = list()
        for iter, seed in enumerate(seeds):
            er, _ = pooling(kwargs, 
			    seed = seed,
			    device = args.device,
                            train_episodes = args.train_episodes,
                            test_episodes = args.test_episodes)
            pool.append(er)
        pool = np.array(pool)
        res = np.mean(pool, axis = 0)
        res_std = np.std(res)
        res_mean = np.mean(res)
        cov = res_std / res_mean  # coefficient of variation
        score = res_mean * (1 - cov)
        print(f'score: {score:.2f} | reward: {res_mean:.2f} +/- {res_std:.2f} | parameters: {kwargs}')
        results.append([score, res_mean, res_std, kwargs])

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
        'learning_rate': [7e-4, 3e-4]
        }
	
    if args.device == 'cuda' and not torch.cuda.is_available():
        print('\nWARNING: GPU not available, switch to CPU\n')
        args.device = 'cpu'
	    
    prime = gridsearch(args, params)
    print(f'\nmaximum score: {prime[0]:.2f} | reward: {prime[1]:.2f} +/- {prime[2]:.2f} | optimal parameters: {prime[3]}')
    print("-------------")


if __name__ == '__main__':
    main()
