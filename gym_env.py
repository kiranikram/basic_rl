# import gym
# import torch.nn as nn
# env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)

# env.action_space.seed(42)

# observation, info = env.reset(seed=42)

# for _ in range(10):
#     observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
#     print('obs',observation)
#     print('rew',reward)


# #replace generating step with linear pytorch:
# obs_dim = env.observation_space
# act_dim = env.action_space

# torch_net = nn.Sequential(
#     nn.Linear(obs_dim,64),
#     nn.Tanh(),
#     nn.Linear(64,64),
#     nn.Tanh(),
#     nn.Linear(64, act_dim)

# )

# env.close()

import argparse
import gym
import numpy as np
from itertools import count
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal


# parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
# parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
#                     help='discount factor (default: 0.99)')
# parser.add_argument('--seed', type=int, default=543, metavar='N',
#                     help='random seed (default: 543)')
# parser.add_argument('--render', action='store_true',
#                     help='render the environment')
# parser.add_argument('--log-interval', type=int, default=10, metavar='N',
#                     help='interval between training status logs (default: 10)')
# args = parser.parse_args()


env = gym.make('CartPole-v1')
env.reset(seed=543)
torch.manual_seed(543)


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, 2)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()
sigma = 0.2
env_action = "discrete" # get from envs 
discount_factor = 0.99
no_of_episodes = 10

#Training code

for episode in range(no_of_episodes):
    log_probs , rewards = [], []

    done = False

    state = env.reset()
    while not done: 
        mu = policy.forward(torch.from_numpy(state).float())
        if env_action == 'continuos':
            distribution = Normal(mu,sigma)
        elif env_action == "discrete":
            action = distribution.sample()

        log_probs.append(distribution.log_prob(action))
        state,reward,done,info = env.step(action.item())

        rewards.append(reward)

cumulative = 0
discounted_r = np.zeros(len(rewards))
for rew in reversed(range(len(rewards))):
    cumulative = cumulative * discount_factor + rewards[rew]
    discounted_r[rew] = cumulative

#normalize the discounted rewards
discounted_r -= np.mean(discounted_r)
discounted_r /= np.std(discounted_r)

#optimization

loss = 0
for i in range(len(rewards)):
    loss += -log_probs[i] * discounted_r[i]
    policy.optimizer.zero_grad()
    loss.backward()
    policy.optimizer.step()



