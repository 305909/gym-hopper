import os
import imageio
import statistics
import numpy as np
import matplotlib.pyplot as plt
import PIL.ImageDraw as ImageDraw

from PIL import Image
from cycler import cycler
from stable_baselines3.common.evaluation import evaluate_policy


class Callback():
    def __init__(self, agent, env, args):
        """ initializes a callback object to access 
        the internal state of the RL agent over training iterations
        
        args:
            agent: reinforcement learning agent
            env: testing environment
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
    
    def _on_step(self, num_episodes, args, verbose = 1) -> bool:
        """ monitors performance """
        if num_episodes % args.eval_frequency == 0: 
            episode_rewards, episode_lengths = evaluate_policy(self.agent, 
                                                               self.env, self.test_episodes, 
                                                               return_episode_rewards = True)
            er, el = np.array(episode_rewards), np.array(episode_lengths)
            self.episode_rewards.append(er.mean())
            self.episode_lengths.append(el.mean())
            if verbose > 0 and num_episodes % int(args.train_episodes * 0.25) == 0:
                print(f'training episode: {num_episodes} | test episodes: {self.test_episodes} | reward: {er.mean():.2f} +/- {er.std():.2f}')
        
        return True


def display(frame, step, episode, reward):
    """ renders the (state -> action) RL agent step in the testing environment 
    as a graphics interchange format frame
    """
    image = Image.fromarray(frame)
    drawer = ImageDraw.Draw(image)
    color = (255, 255, 255) if np.mean(image) < 128 else (0, 0, 0)
    drawer.text((image.size[0] / 20, image.size[1] / 18), 
                f'episode: {episode} | step: {step} | reward: {reward:.2f}', fill = color)
    
    return image


def multiprocess(args, train_env, test_env, train, seeds = [1, 2, 3, 5, 8]):
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


def track(metric, xs, ys, sigmas, args, filename):
    """ plots the RL agent's performance in the testing environment 
    (according to the evaluation metric) over training episodes
    
    args:
        metric: evaluation metric
        xs: training episodes (x-axis values)
        ys: set of evaluation records (y-axis values)
        sigmas: set of evaluation records variances
    """
    colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', 
              '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F', '#BAB0AC']
    plt.rcParams['axes.prop_cycle'] = cycler(color = colors)

    plt.plot(xs, ys, alpha = 1, label = f'A2C')
    plt.fill_between(xs, ys - sigmas, ys + sigmas, alpha = 0.5)
  
    plt.xlabel('episodes')
    plt.ylabel(f'episode {metric}')
    plt.grid(True)
    plt.legend()
  
    plt.savefig(f'{args.directory}/{filename}.png', dpi = 300)
    plt.close()
