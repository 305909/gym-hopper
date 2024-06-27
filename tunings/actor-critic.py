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

from agents.actor_critic import A2C, A2CPolicy
from env.custom_hopper import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-episodes', 
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


def train(device: str = 'cpu', 
          train_episodes: int = 19750, 
          train_env: str = 'CustomHopper-source-v0', **kwargs) -> A2C:
		  
    env = gym.make(train_env)

    """ Training """
		  
    policy = A2CPolicy(env.observation_space.shape[-1], env.action_space.shape[-1], **kwargs)
    agent = A2C(policy, device = device, **kwargs)

    num_episodes = 0
    num_timesteps = 0
    while num_episodes < train_episodes:
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
      
    return agent


def test(agent: A2C, 
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


def pooling(kwargs: dict, device, train_episodes, test_episodes):
    
    agent = train(device = device, 
                  train_episodes = train_episodes, **kwargs)
    
    return test(agent, 
                test_episodes = test_episodes)


def gridsearch(args, params):
    results = []
    keys = list(params.keys())
    for param in itertools.product(*params.values()):
        kwargs = dict(zip(keys, param))
        er = pooling(kwargs, 
                     device = args.device,
                     train_episodes = args.train_episodes,
                     test_episodes = args.test_episodes)
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
    params = {                                           # | source -> source
        'batch_size': [8, 16, 32, 64],                   # | 32
        'learning_rate': [1e-3, 7e-4, 5e-4, 3e-4, 1e-4]  # | 7e-4
        }

    prime = gridsearch(args, params)
    print("---------------------------------------------")
    print(f'Maximum Score: {prime[0]:.2f} | Avg. Reward: {prime[1]:.2f} +/- {prime[2]:.2f} | Optimal Parameters: {prime[3]}')
    print("---------------------------------------------")


if __name__ == '__main__':
    main()
