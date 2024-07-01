import torch
import torch.nn.functional as F

from collections import OrderedDict
from torch.distributions import Normal


class Policy(torch.nn.Module):
    
    def __init__(self, state_space, action_space, actor = True, **kwargs):
        super().__init__()
        
        self.action_space = action_space
        self.state_space = state_space
        self.tanh = torch.nn.Tanh()
        self.actor = actor
        self.hidden = 64

        """ Actor/Critic network """
        self.fc1 = torch.nn.Linear(state_space, self.hidden)
        self.fc2 = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3 = torch.nn.Linear(self.hidden, action_space)

        if actor:
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
        """ Actor/Critic """
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        action_mean = self.fc3(x)

        if self.actor:
            sigma = self.sigma_activation(self.sigma)
            normal_dist = Normal(action_mean, sigma)
            return normal_dist
        else:
            return action_mean

    def to(self, device):
        # move parameters to device
        for param in self.parameters():
            param.data = param.data.to(device)
        return self


class A2CPolicy:
    
    def __init__(self, state_space, action_space, **kwargs):

        self.policies = OrderedDict()
        self.policies['actor'] = Policy(state_space, action_space)
        self.policies['critic'] = Policy(state_space, 1, actor = False)

    def to(self, device):
        # move parameters to device
        for k, v in self.policies.items():
            self.policies[k] = v.to(device)
        return self

    def state_dict(self):
        sd = OrderedDict()
        for k, nn in self.policies.items():
            sd[k] = nn.state_dict()
        return sd

    def load_state_dict(self, states, strict = True):
        for k, sd in states.items():
            self.policies[k].load_state_dict(sd, strict = strict)


class A2C:
    
    def __init__(self, policy, 
                 device: str = 'cpu',
                 learning_rate: float = 1e-4,
                 max_grad_norm: float = 0.5,
                 entropy_coef: float = 0.0,
                 critic_coef: float = 0.5,
                 batch_size: int = 16,
                 gamma: float = 0.99,
                 **kwargs):
                     
        self.device = device
        self.policy = policy.to(self.device)

        self.max_grad_norm = max_grad_norm
        # entropy coefficient to balance exploration/exploitation
        self.entropy_coef = entropy_coef
        # critic coefficient to weight the value function loss
        self.critic_coef = critic_coef

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.gamma = gamma

        self.optimizer = torch.optim.Adam(self.policy.policies['actor'].parameters(), lr = self.learning_rate)
        self.optimizer.add_param_group({'params': self.policy.policies['critic'].parameters()})
                     
        self.reset()

    def predict(self, obs, state = None, episode_start = None, deterministic = False):
        x = torch.from_numpy(obs).float().to(self.device)
        normal_dist = self.policy.policies['actor'](x)

        if deterministic:  # return mean
            action = normal_dist.mean
            action = action.detach().cpu().numpy()
            return action, None
            
        else:  # sample from the distribution
            action = normal_dist.sample()
            action_log_prob = normal_dist.log_prob(action).sum()
            state_value = self.policy.policies['critic'](x)
            action = action.detach().cpu().numpy()
            return action, action_log_prob, state_value

    def store_outcome(self, state, action_log_prob, reward, done, state_value):
        self.states.append(torch.from_numpy(state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.state_values.append(state_value)

    def update_policy(self):
        # stack and move data to device
        states = torch.stack(self.states, dim = 0).to(self.device).squeeze(-1)
        action_log_probs = torch.stack(self.action_log_probs, dim = 0).to(self.device).squeeze(-1)
        rewards = torch.tensor(self.rewards, dtype = torch.float32).to(self.device)
        dones = torch.tensor(self.dones, dtype = torch.float32).to(self.device)
        state_values = torch.stack(self.state_values, dim = 0).to(self.device).squeeze(-1)

        # compute bootstrapped returns
        bootstrapped_returns = torch.zeros_like(rewards)
        cumulative = 0
        for t in reversed(range(len(rewards))):
            cumulative = rewards[t] + self.gamma * cumulative * (1 - dones[t])
            bootstrapped_returns[t] = cumulative

        # compute advantages
        advantages = bootstrapped_returns - state_values
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased = False) + 1e-5)

        # compute actor loss - with entropy regularization to encourage exploration
        entropy = self.policy.policies['actor'](states).entropy().mean()
        actor_loss = - (action_log_probs * advantages.detach()).mean() - self.entropy_coef * entropy

        # compute critic loss
        critic_loss = F.mse_loss(state_values, bootstrapped_returns.detach())

        # compute total loss with value function loss weight
        loss = actor_loss + self.critic_coef * critic_loss
        
        # optimize actor/critic
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.policies['actor'].parameters(), self.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.policy.policies['critic'].parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.reset()

    def reset(self):
        self.states = []
        self.action_log_probs = []
        self.rewards = []
        self.dones = []
        self.state_values = []
