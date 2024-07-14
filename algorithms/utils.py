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


def multiprocess(args, train_env, test_env, train, seeds):
    """ processes multiple sequential training and testing sessions 
    with different seeds (to counteract variance)

    args:
        train_env: training environment
        test_env: testing environment
        train: training function
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


def track(metric, xs, ys, sigmas, args, label, filename):
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

    plt.plot(xs, ys, alpha = 1, label = label)
    plt.fill_between(xs, ys - sigmas, ys + sigmas, alpha = 0.5)
  
    plt.xlabel('episodes')
    plt.ylabel(f'episode {metric}')
    plt.grid(True)
    plt.legend()
  
    plt.savefig(f'{args.directory}/{filename}.png', dpi = 300)
    plt.close()


def init_data(agent, source_env, target_env, masses, num_init = 5, num_roll = 5):
    """ initializes a dataset in which each row: data[row, :-1], data[row, -1] = target, parameters
    
    args: 
        num_init: number of initialization steps
        num_roll: number of rollout to obtain an estimate of the actual value of the unknown objective function
        
    returns: 
        data: dataset of parameters and objective function values
    """

    low = 0.25
    high = 10
    cols = 6

    distribution = Uniform(low = torch.tensor([low], dtype = float), 
                           high = torch.tensor([high], dtype = float))

    data = torch.zeros(num_init, cols + 1)
    
    for i in range(num_init): 
        phi = torch.tensor([distribution.sample() for col in range(cols)], dtype = float)
        data[i, :-1] = phi
        data[i, -1] = J(agent = agent, source_env = source_env, target_env = target_env, 
                        bounds = phi, masses = masses, num_roll = num_roll)
    return data


def J(agent, source_env, target_env, bounds, masses, num_roll):
    source_env.set_parametrization(bounds)
    source_env.set_random_parameters(masses = masses, type = "uniform")

    # learning with respect to random environment
    agent.learn(total_timesteps = int(1e5))

    roll = 0
    roll_rewards = []
    # testing the policy in the target environment for num_roll times
    while roll < num_roll:
        done = False
        obs = env.reset()
        rewards, steps = (0, 0)
        while not done: 
            action, _ = agent.predict(obs)
            next_state, reward, done, _ = env.step(action)
            rewards += reward
            obs = next_state
            # collecting the reward
            test_rewards.append(rewards)
        roll += 1   
        roll_rewards.append(rewards)
        
    rr = np.array(roll_rewards)
    return rr.mean()


def get_candidate(X, Y): 
    gp = SingleTaskGP(X, Y)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_model(mll)
    UCB = UpperConfidenceBound(gp, beta = 0.1, maximize = True)
    bounds = torch.stack([args.min * torch.ones(X.shape[1]), args.max * torch.ones(X.shape[1])])
    candidate, _ = optimize_acqf(UCB, bounds = bounds, q = 1, num_restarts = 5, raw_samples = 20)
    candidate = candidate.reshape(-1,)
    return candidate


def BRN(masses = ["thigh", "leg", "foot"], num_init = 5, num_roll = 5, maxit = 15, verbose = 1): 
    """ exploits bayesian optimization to choose the optimal parametrization from a specific set of parameters
    for what concerns their influence on a some blackbox function
    
    args: 
        num_init: number of initializations iterations to run in order to collect some initial evidence
        num_roll: number of rollout iterations to use so to estimate the actual outcome of some specific set of parameters
        maxit: maximal number of iterations of the overall Bayesian Optimization process.
    """

    # creating source and target environments
    source_env = gym.make("CustomHopper-source-v0")
    target_env = gym.make("CustomHopper-target-v0") 

    # istantiating an agent
    agent = TRPO('MlpPolicy', env = source_env, verbose = verbose)

    data = init_data(agent, source_env, target_env, masses, num_init = num_init, num_roll = num_roll)
    
    for it in tqdm(range(maxit)):  
        X, Y = data[:, :-1], data[:, -1].reshape(-1,1)
        # obtaining best candidate with Bayesian Optimization
        candidate = get_candidate(X, Y)
        # evaluating the candidate solution with source training - rollout evaluation
        phi = J(agent, source_env, target_env, candidate, masses)
        phi = torch.tensor(phi).reshape(-1)

        candidate = torch.hstack([candidate, phi])
        data = torch.vstack((data, candidate))
    
    optimal = D[torch.argmax(data[:, -1]), :-1]
    return data, optimal


def BayRN(masses = ["thigh", "leg", "foot"], verbose = 0):
    data, bc = BRN(masses = masses, verbose = verbose)
    np.savetxt("BayRN_data.txt", D)
    return bc
