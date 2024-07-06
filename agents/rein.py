import torch
import torch.nn.functional as F

from torch.distributions import Normal


class RFPolicy(torch.nn.Module):

    def __init__(self, state_space: int, action_space: int, **kwargs):
        super().__init__()
        """ initializes a multi-layer neural network 
        to map observations from the environment into
        -> elements of a normal distribution (mean and standard deviation) 
           from which to sample an agent action
        
        args:
            state_spaces: dimension of the observation space (environment)
            action_space: dimension of the action space (agent)
        """
        self.action_space = action_space
        self.state_space = state_space
        self.tanh = torch.nn.Tanh()
        self.hidden = 64
        self.eps = 1e-6

        """ policy mean specific layers """
        self.fc1 = torch.nn.Linear(state_space, self.hidden)
        self.fc2 = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3 = torch.nn.Linear(self.hidden, action_space)

        """ policy standard deviation specific layer """
        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space) + init_sigma)
        self.sigma_activation = F.softplus

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        """ maps the observation x from the environment into 
        -> a normal distribution N(μ,σ) from which to sample an agent action
        
        args:
            x: observation from the environment

        returns:
            normal_dist: normal distribution N(μ,σ)
                         - μ: action mean
                         - σ: action standard deviation
        """
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        action_mean = self.fc3(x)
        
        sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(action_mean + self.eps, sigma + self.eps)
        return normal_dist

    def to(self, device):
        """ move parameters to device """
        for param in self.parameters():
            param.data = param.data.to(device)
        return self


class RF:
    
    def __init__(self, policy, 
                 device: str = 'cpu', 
                 baseline: str = 'vanilla',
                 learning_rate: float = 1e-3, 
                 max_grad_norm: float = 0.5, 
                 gamma: float = 0.99,  
                 **kwargs):
        """ initializes an agent that learns a policy via REINFORCE algorithm 
        to solve the task at hand (Gym Hopper)

        args:
            policy: RFPolicy
            device: processing device (e.g. CPU or GPU)
            baseline: algorithm baseline
            learning_rate: learning rate for policy optimization
            max_grad_norm: threshold value for the gradient norm
            gamma: discount factor
        """
        self.device = device
        self.policy = policy.to(self.device)
        self.max_grad_norm = max_grad_norm
                     
        self.learning_rate = learning_rate
        self.baseline = baseline
        self.gamma = gamma

        self.optimizer = torch.optim.Adam(policy.parameters(), lr = self.learning_rate)
                     
        self.reset()

    def predict(self, obs, state = None, episode_start = None, deterministic = False):
        """ predicts an action, according to 
        -> the policy
        -> the observation

        args:
            obs: observation from the environment

        returns:
            action: action to perform
            action_log_prob: logarithmic probability value of the action
        """
        x = torch.from_numpy(obs).float().to(self.device)
        normal_dist = self.policy(x)

        if deterministic:
            """ return the mean 
            of a normal distribution N(μ,σ)
            
            returns:
                a = μ
            """
            action = normal_dist.mean
            action = action.detach().cpu().numpy()
            return action, None
            
        else:
            """ sample an action 
            from a normal distribution N(μ,σ)

            returns:
                a ~ N(μ,σ)
            """
            action = normal_dist.sample()
            action_log_prob = normal_dist.log_prob(action).sum()
            action = action.detach().cpu().numpy()
            return action, action_log_prob

    def store_outcome(self, state, action_log_prob, reward):
        self.states.append(torch.from_numpy(state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(reward)
        
    def update_policy(self):
        """ stack and move data to device """
        states = torch.stack(self.states, dim = 0).to(self.device).squeeze(-1)
        action_log_probs = torch.stack(self.action_log_probs, dim = 0).to(self.device).squeeze(-1)
        rewards = torch.tensor(self.rewards, dtype = torch.float32).to(self.device)
        
        """ compute discounted returns (backwards) """
        discounted_returns = torch.zeros_like(rewards)
        cumulative = 0
        for t in reversed(range(len(rewards))):
            cumulative = rewards[t] + self.gamma * cumulative
            discounted_returns[t] = cumulative

        """ enforce baseline """
        if self.baseline == 'constant':
            discounted_returns -= 20
        if self.baseline == 'whitening':
            discounted_returns = (discounted_returns - discounted_returns.mean()) / discounted_returns.std()

        """ compute actor loss """
        loss = - (action_log_probs * discounted_returns).mean()

        """ updates the policy network's weights """
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.reset()

    def reset(self):
        self.states = []
        self.action_log_probs = []
        self.rewards = []
