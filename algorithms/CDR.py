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

from stable_baselines3 import SAC
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
    parser.add_argument('--train-episodes', default = 10000, type = int, 
                        help = 'number of training episodes')
    parser.add_argument('--test-episodes', default = 50, type = int, 
                        help = 'number of testing episodes')
    parser.add_argument('--eval-frequency', default = 100, type = int, 
                        help = 'evaluation frequency over training iterations')
    parser.add_argument('--model', default = 'PPO', type = str, choices = ['SAC', 'PPO'],
                        help = 'reinforcement learning algorithm (SAC or PPO)')
    parser.add_argument('--directory', default = 'results', type = str, 
                        help = 'path to the output location for checkpoint storage (model and rendering)')
    return parser.parse_args()


def get(model):
    if model == 'SAC': return SAC
    elif model == 'PPO': return PPO
    else: raise ValueError(f"ERROR: model: {model} not found")


class ADR():
	def __init__(self, init_params: dict, p_b = 0.5, m = 50, delta = 0.2, step = 'constant', thresholds:list = [1500, 1700]) -> None:
		self.init_params = init_params # dict of the form {'torso': val, 'leg': val, 'foot': val}
		self.thresholds = thresholds
		self.delta = delta
		self.m = m  # data buffer size 
		self.bounds = self._init_bounds() # bounds: dict {'torso_low': val, 'torso_high': val, ...}
		self.p_b = p_b
		self.step = getattr(self, '_' + step)
		self.thigh_mass = None
		self.leg_mass = None
		self.foot_mass = None
		self.rewards = []
		self.weights = []
		self.current_weight = np.float64(1)
		self.last_performances = []
		self.last_increments = []
		self.part = ['thigh', 'leg', 'foot']
		self.databuffer = {"thigh_low": [], 
                       "thigh_high": [],
                       "leg_low": [], 
                       "leg_high": [],
                       "foot_low": [],
                       "foot_high": []
                      }
		self.keys = list(self.databuffer.keys())
		self.current_bound = 0
    
	def insert(self, body_part: str, reward: float) -> None:
		if self.keys[self.current_bound] == body_part:
			self.databuffer[body_part].append(reward)

	# compute the mean performance and clear buffer
	def _evaluate_perfomance(self, body_part: str) -> float:
		performance = np.mean(np.array(self.databuffer[body_part]))
		self.databuffer[body_part].clear()
		return performance

	# Check size of the ADR and in case increase or decrease the bounds
	def updateADR(self, body_part: str):
		if len(self.databuffer[body_part]) >= self.m:
			# low or high
			bp, extract_extreme = tuple(body_part.split("_"))
			performance = self._evaluate_perfomance(body_part)
			
			self.last_performances.append((body_part, performance))

			# if performance >= self._th('high'):
			if performance >= self.thresholds[1]:
				if extract_extreme == "high":
					self._increase_high_bounds(body_part, performance)
				else:
					self._decrease_low_bounds(body_part, performance)
			# elif performance <= self._th('low'):
			if performance <= self.thresholds[0]:
				if extract_extreme == "high":
					self._decrease_high_bounds(body_part, performance)
				else:
					self._increase_low_bounds(body_part, performance)

	def get_random_masses(self):
		# Set three random masses
		thigh_mass = np.random.uniform(
			self.bounds["thigh_low"], self.bounds["thigh_high"])
		leg_mass = np.random.uniform(
			self.bounds["leg_low"], self.bounds["leg_high"])
		foot_mass = np.random.uniform(
			self.bounds["foot_low"], self.bounds["foot_high"])

		d = {"thigh": thigh_mass, "leg": leg_mass, "foot": foot_mass}

		# prob of set masses to lower or upper bound
		u = np.random.uniform(0, 1)
		k_bounds = None

		# Set one random parameter to its lower or upper bound
		if u < self.p_b:
			k_bounds = self._select_random_parameter()
			body_part = k_bounds.split("_")[0]

			d[body_part] = self.bounds[k_bounds]

		return list(d.values()), k_bounds

	def evaluate(self, episode_return, key_bounds) -> None:
		self.insert_ep_return(body_part=key_bounds, ep_return=episode_return)
		self.updateADR(body_part=key_bounds)

	def _constant(self, *args):
		return self.delta

	def _increase_high_bounds(self, body_part: str, performance):
		step = self.step(self.thresholds[1], performance)
		self.bounds[body_part] = self.bounds[body_part] + step 
		self.last_increments.append((body_part, 'high+', step))
		self.current_bound = (self.current_bound+1) % len(self.keys)
		# if self.update_counter == 0:
		# 	self.current_bound = (self.current_bound+1) % len(self.keys)
		# else:
		# 	self.update_counter += 1

	def _decrease_low_bounds(self, body_part: str, performance):
		step = self.step(self.thresholds[1], performance)
		new_low_bounds = self.bounds[body_part] - step
		self.bounds[body_part] = max(new_low_bounds, 0)
		self.last_increments.append((body_part, 'low+', step))
		self.current_bound = (self.current_bound+1) % len(self.keys)
		# if self.update_counter == 0:
		# 	self.current_bound = (self.current_bound+1) % len(self.keys)
		# else:
		# 	self.update_counter += 1
	
	def _decrease_high_bounds(self, body_part: str, performance):
		body = body_part.split('_')[0]
		if not np.isclose(self.init_params[body], self.bounds[body_part]):
			# self.update_counter -= 1
			self.bounds[body_part] = max(self.bounds[body_part] - self.delta, self.init_params[body])
		self.last_increments.append((body_part, 'high-', self.delta))
	
	def _increase_low_bounds(self, body_part: str, performance):
		body = body_part.split('_')[0]
		if not np.isclose(self.init_params[body], self.bounds[body_part]):
			# self.update_counter -= 1
			self.bounds[body_part] = min(self.bounds[body_part] + self.delta, self.init_params[body])
		self.last_increments.append((body_part, 'low-', self.delta))

	def _init_bounds(self):
		try:
			dict = {"thigh_low": self.init_params['thigh'],
					"thigh_high": self.init_params['thigh'],
					"leg_low": self.init_params['leg'],
					"leg_high": self.init_params['leg'],
					"foot_low": self.init_params['foot'],
					"foot_high": self.init_params['foot']
					}
		except:
			print("Bounds not initialized")
		return dict

	# Extract random key
	def _select_random_parameter(self) -> str:
		# keys = list(self.bounds.keys())
		# rand = np.random.choice(len(keys))
		rand = np.random.randint(2) # random 0 or 1, if 1 changes the bound of the parameter that we are testing
		part = self.keys[self.current_bound^rand]
		# print('selected bound:', part) # TODO: remove this print
		return part
		# return keys[rand]
  
	def print_distributions(self):
		for p in self.part:
			high = self.bounds[p+'_high']
			low = self.bounds[p+'_low']
			center = self.init_params[p]
			left = round((center - low) // self.delta)
			right = round((high - center) // self.delta)
			if low != center:
				print(f'\t[{round(low, 2)}]«{"-"*left}', end='')
			else:
				print('\t', end='')
			print(f'[{round(center, 2)}]', end='')
			if high != center:
				print(f'{"-"*right}»[{round(high, 2)}]')
			else:
				print()
        

class Callback(BaseCallback):
    def __init__(self, agent, env, auto, ADR, args, verbose = 1):
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
        self.mod = args.model
        self.original_masses = {'thigh': 3.9269908169872427, 
                                'foot': 5.0893800988154645,
                                'leg': 2.7143360527015816}
        self.masses = {'thigh': [(3.9269908169872427, 3.9269908169872427)],  
                       'foot': [(5.0893800988154645, 5.0893800988154645)], 
                       'leg': [(2.7143360527015816, 2.7143360527015816)]}
        self.agent = agent
        self.bounds = None
        self.flag = False
        self.auto = auto
        self.adr = ADR
        self.env = env
    
    def _on_step(self) -> bool:
        """ monitors performance """
        self.num_episodes += np.sum(self.locals['dones'])
        if (self.locals['dones']):
            if self.bounds is not None:
                self.adr.evaluate(infos['episode']['r'], bounds)
            params, self.bounds = self.adr.get_random_masses()
            print(params, self.bounds)
            print(self.auto.sim.model.body_mass[1:], self.auto..sim.model.body_mass[2:])
            self.auto.set_parameters(params)
            self.adr.print_distributions()
      
        if self.num_episodes % self.eval_frequency == 0: 
            if not self.flag:
                episode_rewards, episode_lengths = evaluate_policy(self.agent, 
                                                                   self.env, self.test_episodes, 
                                                                   return_episode_rewards = True)
                er, el = np.array(episode_rewards), np.array(episode_lengths)
                self.episode_rewards.append(er.mean())
                self.episode_lengths.append(el.mean())
              
                if self.verbose > 0 and self.num_episodes % int(self.train_episodes * 0.25) == 0:
                    print(f'training episode: {self.num_episodes} | test episodes: {self.test_episodes} | reward: {er.mean():.2f} +/- {er.std():.2f} | bounds: ({self.auto.data_buffers[self.mod]["L"][self.auto.i - 1]:.2f}, {self.auto.data_buffers[self.mod]["H"][self.auto.i - 1]:.2f}) | -> phi: {self.auto.phi:.2f}')
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
    model = args.model

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
    MOD = get(model)
	
    if model == 'SAC':
        agent = MOD(policy, 
                    env = env, 
                    seed = seed,
                    device = args.device, 
                    learning_rate = 7.5e-4,
                    batch_size = 256, 
                    gamma = 0.99)
	    
    elif model == 'PPO':
        agent = MOD(policy, 
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

    init_params = {"thigh": 3.92699082,  "leg": 2.71433605, "foot": 5.0893801}
    ADRCallback = ADR(init_params)
  
    callback = Callback(agent, test_env, env, ADRCallback, args)

    total_timesteps = args.train_episodes * 500
    start_time = time.time()
    agent.learn(total_timesteps = total_timesteps, callback = callback)
    train_time = time.time() - start_time
    
    return callback.episode_rewards, callback.episode_lengths, callback.masses, train_time, agent.policy.state_dict()


def test(args, test_env):
    """ tests the agent in the testing environment """
    env = gym.make(test_env)
    policy = 'MlpPolicy'
    model = args.model
    MOD = get(model)

    path = f'{args.directory}/{model}-ADR.mdl'
    agent = MOD.load(path, env = env, device = args.device)
    
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
        imageio.mimwrite(f'{args.directory}/{model}-ADR-test.gif', frames, fps = 30)

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
    model = args.model
    MOD = get(model)
	
    if model == 'SAC':
        agent = MOD(policy, 
                    env = env, 
                    device = args.device, 
                    learning_rate = 7.5e-4,
                    batch_size = 256, 
                    gamma = 0.99)
	    
    elif model == 'PPO':
        agent = MOD(policy, 
                    env = env, 
                    device = args.device, 
                    learning_rate = 2.5e-4,
                    batch_size = 128, 
		    ent_coef = 0.0,
                    n_steps = 1024,
                    gamma = 0.99)
        
    agent.policy.load_state_dict(weights)
    agent.save(f'{args.directory}/{model}-ADR.mdl')
    print(f'\nmodel checkpoint storage: {args.directory}/{model}-ADR.mdl\n')


def plot(args, records):
    # initialize dictionaries
    stacks = {key: [] for key in records[0]}
    
    # calculate averages for each key across each record
    for key in records[0]:
        for i in range(len(records[0][key])):
            # calculate average of the first and second elements separately
            lower = np.mean([record[key][i][0] for record in records])
            upper = np.mean([record[key][i][1] for record in records])
            stacks[key].append((lower, upper))
    
    # stack averages into xs and ys arrays
    xs = np.array([(index + 1) * args.eval_frequency for index in range(len(stacks[key]))])
    ys = dict()
    for key in records[0]:
        ys[key] = stacks[key]
	
    colors = ['#4E79A7', '#E15759', '#59A14F']
    plt.rcParams['axes.prop_cycle'] = cycler(color = colors)
	
    # iterate over each key and plot lower and upper values separately
    for index, (key, values) in enumerate(ys.items()):
        lowers = [y[0] for y in values]
        uppers = [y[1] for y in values]
        
        # plot lower values
        plt.plot(xs, lowers, alpha = 1, 
		 label = f'{key}', color = colors[index % len(colors)])

        path = os.path.join(args.directory, f'{args.model}-ADR-masses-{key}-lowers.npy')
        np.save(path, lowers)
	    
        # plot upper values
        plt.plot(xs, uppers, alpha = 1, 
		 color = colors[index % len(colors)])

        path = os.path.join(args.directory, f'{args.model}-ADR-masses-{key}-uppers.npy')
        np.save(path, uppers)

    plt.xlabel('episodes')
    plt.ylabel(f'mass (kg)')
    plt.grid(True)
    plt.legend()
	
    plt.savefig(f'{args.directory}/{args.model}-ADR-masses.png', dpi = 300)
    plt.close()


def main():
    args = parse_args()
    warnings.filterwarnings("ignore")

    if not os.path.exists(args.directory):
        os.mkdir(args.directory)

    train_env, test_env = tuple(f'CustomHopper-{x}-v0' 
                                for x in ['source-ADR', 'target'])

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
	    
    if args.model == 'SAC':
        args.train_episodes = 1000
        args.eval_frequency = 10
	    
    if args.train:
        pool = multiprocess(args, train_env, test_env, train)
        for metric, records in zip(('reward', 'length'), (pool['rewards'], pool['lengths'])):
            metric, xs, ys, sigmas = stack(args, metric, records)
            if metric == 'reward':
                path = os.path.join(args.directory, f'{args.model}-ADR-rewards.npy')
                np.save(path, ys)
            track(metric, xs, ys, sigmas, args, 
                  label = f'{args.model}-ADR', 
                  filename = f'{args.model}-ADR-{metric}')
            plot(args, records = pool['masses'])
        print(f'\ntraining time: {np.mean(pool["times"]):.2f} +/- {np.std(pool["times"]):.2f}')
        print("-------------")

        arrange(args, pool['weights'], train_env)
        
    if args.test:
        test(args, test_env)


if __name__ == '__main__':
    main()
