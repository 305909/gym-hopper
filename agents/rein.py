import torch
import torch.nn.functional as F

from torch.distributions import Normal


class RFPolicy(torch.nn.Module):

    def __init__(self, state_space, action_space, **kwargs):
        super().__init__()
        
        self.action_space = action_space
        self.state_space = state_space
        self.tanh = torch.nn.Tanh()
        self.hidden = 64

        """ Actor network """
        self.fc1 = torch.nn.Linear(state_space, self.hidden)
        self.fc2 = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3 = torch.nn.Linear(self.hidden, action_space)

        # standard deviation for exploration at training time
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
        """ Actor """
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        action_mean = self.fc3(x)
        
        sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(action_mean, sigma)
        return normal_dist

    def to(self, device):
        # move parameters to device
        for param in self.parameters():
            param.data = param.data.to(device)
        return self


class RF:
    
    def __init__(self, policy, 
                 device: str = 'cpu', 
                 baseline: str = 'vanilla',
                 learning_rate: float = 1e-4, 
                 max_grad_norm: float = 0.5, 
                 entropy_coef: float = 0.0, 
                 gamma: float = 0.99,  
                 **kwargs):
   
        self.device = device
        self.policy = policy.to(self.device)

        # clipping coefficient to clip the gradient
        self.max_grad_norm = max_grad_norm
        # entropy coefficient to balance exploration/exploitation
        self.entropy_coef = entropy_coef

        self.learning_rate = learning_rate
        self.baseline = baseline
        self.gamma = gamma

        self.optimizer = torch.optim.Adam(policy.parameters(), lr = self.learning_rate)
                     
        self.reset()

    def predict(self, obs, state = None, episode_start = None, deterministic = False):
        x = torch.from_numpy(obs).float().to(self.device)
        normal_dist = self.policy(x)

        if deterministic:  # return mean
            action = normal_dist.mean
            action = action.detach().cpu().numpy()
            return action, None
            
        else:  # sample from the distribution
            action = normal_dist.sample()
            action_log_prob = normal_dist.log_prob(action).sum()
            action = action.detach().cpu().numpy()
            return action, action_log_prob

    def store_outcome(self, state, action_log_prob, reward):
        self.states.append(torch.from_numpy(state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(reward)
        
    def update_policy(self):
        # stack and move data to device
        states = torch.stack(self.states, dim = 0).to(self.device).squeeze(-1)
        action_log_probs = torch.stack(self.action_log_probs, dim = 0).to(self.device).squeeze(-1)
        rewards = torch.tensor(self.rewards, dtype = torch.float32).to(self.device)
        
        # compute discounted returns
        discounted_returns = torch.zeros_like(rewards)
        cumulative = 0
        for t in reversed(range(len(rewards))):
            cumulative = rewards[t] + self.gamma * cumulative
            discounted_returns[t] = cumulative

        # enforce baseline
        if self.baseline == 'constant':
            discounted_returns -= 20
        if self.baseline == 'whitening':
            discounted_returns = (discounted_returns - discounted_returns.mean()) / discounted_returns.std()

        # compute actor loss - with entropy regularization to encourage exploration
        entropy = self.policy(states).entropy().mean()
        loss = - (action_log_probs * discounted_returns).mean() - self.entropy_coef * entropy

        # optimize actor
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.reset()

    def reset(self):
        self.states = []
        self.action_log_probs = []
        self.rewards = []
