import torch
import torch.nn as nn
import numpy as np


def compute_gen_adv_est(
    rewards: list,
    values: list,
    is_done: list,
    gamma: float,
    lamda: float,
):
    """
    Returns Generalized Advantage Estimate that can be used as a baseline in the update.

    """
    adv_est = 0
    gaes = []
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1] * is_done[i] - values[i]
        adv_est = delta + gamma * lamda * is_done[i] * adv_est
        gaes.insert(0, adv_est + values[i])

    return gaes


def get_trajectory(
    states: torch.Tensor,
    actions: torch.Tensor,
    returns: torch.Tensor,
    log_probs: torch.Tensor,
    values: torch.Tensor,
    advantages: torch.Tensor,
    batch_size,
    num_epochs,
):
    """
    Generates dataset of trajectories for usage in policy update.
    """
    data_len = states.size(0)
    for _ in range(num_epochs):
        for _ in range(data_len // batch_size):
            ids = np.random.choice(data_len, batch_size)
            yield states[ids, :], actions[ids], returns[ids], log_probs[ids], values[
                ids
            ], advantages[ids]


def orthogonal_init(m):
    """As stated in "https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/" , orthogonal initialization of weights outperforms Xavier initilization"""
    if type(m) in (nn.Linear, nn.Conv2d):
        nn.init.orthogonal_(m.weight.data, np.sqrt(float(2)))
        if m.bias is not None:
            m.bias.data.fill_(0)
