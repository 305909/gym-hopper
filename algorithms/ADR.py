""" ADR algorithm
Custom Hopper 
MuJoCo environment
"""

import os
import gym
import sys
import time
import torch
import imageio
import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt

import stable_baselines3

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..')))

from PIL import Image
from cycler import cycler
from env.custom_hopper import *

from stable_baselines3 import PPO
from collections import OrderedDict
from utils import display, stack, track
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
    parser.add_argument('--train-episodes', default = 10000, type = int,
                        help='number of training episodes')
    parser.add_argument('--test-episodes', default = 50, type = int,
                        help = 'number of testing episodes')
    parser.add_argument('--eval-frequency', default = 100, type = int,
                        help = 'evaluation frequency over training iterations')
    parser.add_argument('--directory', default = 'results', type = str,
                        help = 'path to the output location for checkpoint storage (model and rendering)')
    return parser.parse_args()


class ADR():
    def __init__(self, params: dict, prob = 0.5, m = 50, delta = 0.2, step = 'constant', thresholds: list = [250, 1750]) -> None:
        self.step = getattr(self, '_' + step)
        self.bounds = {"thigh_low": params['thigh'],
		       "thigh_high": params['thigh'],
		       "leg_low": params['leg'],
		       "leg_high": params['leg'],
		       "foot_low": params['foot'],
		       "foot_high": params['foot']}
        self.thresholds = thresholds
        self.thigh_mass = None
        self.foot_mass = None
        self.leg_mass = None
        self.params = params
        self.delta = delta
        self.prob = prob
        self.m = m
        self.current_weight = np.float64(1)
        self.last_performances = []
        self.last_increments = []
        self.rewards = []
        self.weights = []
        self.part = ['thigh', 'leg', 'foot']
        self.databuffer = {"thigh_low": list(),
                           "thigh_high": list(),
                           "leg_low": list(),
                           "leg_high": list(),
                           "foot_low": list(),
                           "foot_high": list()}
        self.keys = list(self.databuffer.keys())
        self.bound = 0

    def insert(self, part: str, reward: float) -> None:
        if self.keys[self.bound] == part:
            self.databuffer[part].append(reward)

    # compute the mean performance and clear buffer
    def evaluate_performance(self, part: str) -> float:
        performance = np.mean(np.array(self.databuffer[part]))
        self.databuffer[part].clear()
        return performance

    # check size of the ADR and in case increase or decrease the bounds
    def update(self, part: str):
        if len(self.databuffer[part]) >= self.m:
            # low or high
            _, extreme = tuple(part.split("_"))
            performance = self.evaluate_performance(part)
            self.last_performances.append((part, performance))

            if performance >= self.thresholds[1]:
                if extreme == "high":
                    self._increase_high_bounds(part, performance)
                else:
                    self._decrease_low_bounds(part, performance)
            if performance <= self.thresholds[0]:
                if extreme == "high":
                    self._decrease_high_bounds(part, performance)
                else:
                    self._increase_low_bounds(part, performance)

    def get_random_masses(self):
        # set three random masses
        thigh_mass = np.random.uniform(
            self.bounds["thigh_low"], self.bounds["thigh_high"])
        leg_mass = np.random.uniform(
            self.bounds["leg_low"], self.bounds["leg_high"])
        foot_mass = np.random.uniform(
            self.bounds["foot_low"], self.bounds["foot_high"])

        dict = {"thigh": thigh_mass, "leg": leg_mass, "foot": foot_mass}

        # probability to set masses to lower or upper bound
        uniform = np.random.uniform(0, 1)
        k = None

        # set one random parameter to its lower or upper bound
        if uniform < self.prob:
            k = self._select_random_parameter()
            part = k.split("_")[0]
            dict[part] = self.bounds[k]

        return list(dict.values()), k

    def evaluate(self, reward, key) -> None:
        self.insert(part = key, reward = reward)
        self.update(part = key)

    def _constant(self, *args):
        return self.delta

    def _increase_high_bounds(self, part: str, performance):
        step = self.step(self.thresholds[1], performance)
        self.bounds[part] = self.bounds[part] + step
        self.bound = (self.bound + 1) % len(self.keys)

    def _decrease_low_bounds(self, part: str, performance):
        step = self.step(self.thresholds[1], performance)
        self.bounds[part] = max(self.bounds[part] - step, 0)
        self.bound = (self.bound + 1) % len(self.keys)

    def _decrease_high_bounds(self, part: str, performance):
        body = part.split('_')[0]
        if not np.isclose(self.params[body], self.bounds[part]):
            self.bounds[part] = max(self.bounds[part] - self.delta, self.params[body])

    def _increase_low_bounds(self, part: str, performance):
        body = part.split('_')[0]
        if not np.isclose(self.params[body], self.bounds[part]):
            self.bounds[part] = min(self.bounds[part] + self.delta, self.params[body])

    # extract random key
    def _select_random_parameter(self) -> str:
        rand = np.random.randint(2)
        part = self.keys[self.bound ^ rand]
        return part


class Callback(BaseCallback):
    def __init__(self, agent, env, auto, adr, args, verbose = 1):
        super(Callback, self).__init__(verbose)
        """ initializes a callback object to access 
        the internal state of the RL agent over training iterations

        args:
            agent: reinforcement learning agent
            env: testing environment for performance evaluation
            auto: training environment with automatic domain randomization (ADR)
            args: argument parser

        evaluation metrics:
            episode rewards
            episode lengths
        """
        self.train_episodes = args.train_episodes
        self.eval_frequency = args.eval_frequency
        self.test_episodes = args.test_episodes
        self.episode_rewards = list()
        self.episode_lengths = list()
        self.num_episodes = 0
        self.masses = {'thigh': [3.9269908169872427],
                       'leg': [2.7143360527015816],
                       'foot': [5.0893800988154645]}
        self.agent = agent
        self.bounds = None
        self.flag = False
        self.auto = auto
        self.adr = adr
        self.env = env
        self.m = 'PPO'
    
    def _on_step(self) -> bool:
        """ monitors performance """
        self.num_episodes += np.sum(self.locals['dones'])
        if self.locals['dones']:
            if self.bounds is not None:
                for info in self.locals['infos']:
                    if 'episode' in info:
                        self.adr.evaluate(info['episode']['r'], self.bounds)
            params, self.bounds = self.adr.get_random_masses()
            self.auto.sim.model.body_mass[2:] = np.array(params)
      
        if self.num_episodes % self.eval_frequency == 0: 
            if not self.flag:
                episode_rewards, episode_lengths = evaluate_policy(self.agent, 
                                                                   self.env, self.test_episodes, 
                                                                   return_episode_rewards = True)
                er, el = np.array(episode_rewards), np.array(episode_lengths)
                self.episode_rewards.append(er.mean())
                self.episode_lengths.append(el.mean())
		    
                masses = self.auto.get_parameters()[1:]
                for i, key in enumerate(self.masses.keys()):
                    self.masses[key].append(masses[i])
			
                if self.verbose > 0 and self.num_episodes % int(self.train_episodes * 0.25) == 0:
                    print(f'training episode: {self.num_episodes} | test episodes: {self.test_episodes} | reward: {er.mean():.2f} +/- {er.std():.2f}')
                self.flag = True  # mark evaluation as due
        if self.num_episodes % self.eval_frequency != 0:
            self.flag = False  # reset evaluation flag
          
        if self.num_episodes >= self.train_episodes: 
            return False  # stop training
          
        return True


def multiprocess(args, train_env, test_env, train, seeds = [1, 2, 3]):
    """ processes multiple sequential training and testing sessions 
    with different seeds (to counteract variance)

    args:
        train_env: training environment
        test_env: testing environment
        train: training function
    """
    pool = {'rewards': list(), 'lengths': list(), 'masses': list(), 'times': list(), 'weights': list()}
    for iter, seed in enumerate(seeds):
        print(f'\ntraining session: {iter + 1}')
        print("----------------")
        for key, value in zip(pool.keys(), train(args, seed, train_env, test_env)):
            pool[key].append(value)
    
    return pool


def train(args, seed, train_env, test_env):
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
    
    agent = PPO(policy, 
                env = env, 
                seed = seed,
                device = args.device, 
                learning_rate = 2.5e-4,
                batch_size = 128, 
                ent_coef = 0.0,
                n_steps = 1024,
                gamma = 0.99)

    test_env = gym.make(test_env)
    test_env.seed(seed)

    params = {"thigh": 3.92699082,  "leg": 2.71433605, "foot": 5.0893801}
    adr = ADR(params)
  
    callback = Callback(agent, test_env, env, adr, args)

    total_timesteps = args.train_episodes * 500
    start_time = time.time()
    agent.learn(total_timesteps = total_timesteps, callback = callback)
    train_time = time.time() - start_time
    
    return callback.episode_rewards, callback.episode_lengths, callback.masses, train_time, agent.policy.state_dict()


def test(args, test_env):
    """ tests the agent in the testing environment """
    env = gym.make(test_env)
    policy = 'MlpPolicy'
	
    path = f'{args.directory}/PPO-ADR.mdl'
    agent = PPO.load(path, env = env, device = args.device)

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
        imageio.mimwrite(f'{args.directory}/PPO-ADR-test.gif', frames, fps = 30)

    env.close()


def plot(args, records):
    # initialize dictionaries
    stacks = {key: [] for key in records[0]}
    
    # calculate averages for each key across each record
    for key in records[0]:
        for i in range(len(records[0][key])):
            value = np.mean([record[key][i] for record in records])
            stacks[key].append(value)
    
    # stack averages into xs and ys arrays
    xs = np.array([(index + 1) * args.eval_frequency for index in range(len(stacks[key]))])
    ys = dict()
    for key in records[0]:
        ys[key] = stacks[key]
	
    colors = ['#4E79A7', '#E15759', '#59A14F']
    plt.rcParams['axes.prop_cycle'] = cycler(color = colors)
	
    # iterate over each key and plot lower and upper values separately
    for index, (key, values) in enumerate(ys.items()):
        masses = [y for y in values]
        
        plt.plot(xs, masses, alpha = 1,
		 label = f'{key}', color = colors[index % len(colors)])

        path = os.path.join(args.directory, f'PPO-ADR-masses-{key}-lowers.npy')
        np.save(path, masses)

    plt.xlabel('episodes')
    plt.ylabel(f'mass (kg)')
    plt.grid(True)
    plt.legend()
	
    plt.savefig(f'{args.directory}/PPO-ADR-masses.png', dpi = 300)
    plt.close()


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
	    
    if args.train:
        pool = multiprocess(args, train_env, test_env, train)
        for metric, records in zip(('reward', 'length'), (pool['rewards'], pool['lengths'])):
            metric, xs, ys, sigmas = stack(args, metric, records)
            if metric == 'reward':
                path = os.path.join(args.directory, f'PPO-ADR-rewards.npy')
                np.save(path, ys)
            track(metric, xs, ys, sigmas, args, 
                  label = f'PPO-ADR', 
                  filename = f'PPO-ADR-{metric}')
        plot(args, records = pool['masses'])
        print(f'\ntraining time: {np.mean(pool["times"]):.2f} +/- {np.std(pool["times"]):.2f}')
        print("-------------")

        policy = 'MlpPolicy'
        agent = PPO(policy, env = env, device = args.device)
        
        agent.policy.load_state_dict(pool['weights'][0])
        agent.save(f'{args.directory}/PPO-ADR.mdl')
        print(f'\nmodel checkpoint storage: {args.directory}/PPO-ADR.mdl\n')
        
    if args.test:
        test(args, test_env)


if __name__ == '__main__':
    main()
