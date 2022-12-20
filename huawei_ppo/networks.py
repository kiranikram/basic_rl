import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

class ActorNetwork(nn.Module):
    def __init__(
        self,obs_dim: int,hidden_size:int,act_dim: int):
        
        super(ActorNetwork, self).__init__()
        self.hidden1 = nn.Linear(obs_dim, hidden_size)
        self.hidden2 = nn.Linear(hidden_size, act_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.hidden1(state))
        x = self.hidden2(x)
        probs = F.softmax(x, dim=1)

        dist = Categorical(probs)
        action = dist.sample()
        return action, dist
        
        
class CriticNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden_size:int):
        
        super(CriticNetwork, self).__init__()

        self.hidden = nn.Linear(obs_dim, hidden_size)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.hidden(state))
        value = self.out(x)

        return value

def orthogonal_init(m):
    """As stated in "https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/" , orthogonal initialization of weights outperforms Xavier initilization"""
    if type(m) in (nn.Linear, nn.Conv2d):
        nn.init.orthogonal_(m.weight.data, np.sqrt(float(2)))
        if m.bias is not None:
            m.bias.data.fill_(0)