import torch
import pandas as pd
import torch.optim as optim
from torch.distributions import Normal, Categorical
import numpy as np
import gym
import plotnine as gg
from helpers import compute_gen_adv_est, get_trajectory ,orthogonal_init
from networks import ActorNetwork , CriticNetwork

class History:
    """"Structure to hold in memory relevant trajectory information as well as replace trajectory information as required"""
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.is_done = []
        self.log_probs = []
        self.values = []

    def clear_history(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.is_done = []
        self.log_probs = []
        self.values = []


class MainAgent(object):
    """
    Params:
        
        curr_env: RL env
        obs_dim: state space
        act_dim: action space
        gamma: discount factor
        lamda: for td lambda in gae
        entropy_coef: entropy factor policy update
        epsilon: clipping parameter for ppo
        actor_lr: actor network learning rate
        critic_lr: critic network learning rate
        range: clipping param
        timesteps: timesteps per trajectory
        samples: no_sample_trajectories
        num_epochs: iteration steps
        batch_size: batch size
        target_kl: Check Kullback-Leibler (KL) divergence does not go too high 


        Returns:
        As stated in "https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/", 
        in addition to providing
        score / reward metrics, seperate actor and critic 
        plots to compare convergence properties. Train 
        functoin return plots for all three. 
        
        """
    def __init__(
        self,
        curr_env,
        gamma: float,
        lamda: float,
        entropy_coef: float,
        epsilon: float,
        range: float,
        timesteps: int,
        samples: int,
        num_epochs: int,
        batch_size: int,
        target_kl : int,
        actor_lr: float = 1e-4,
        critic_lr:float = 5e-4,
        
        ):
       
        
        self.env = curr_env
        self.env.reset(seed=543)
        torch.manual_seed(543)

        
        self.gamma = gamma
        self.lamda = lamda
        self.entropy_coef = entropy_coef
        self.epsilon = epsilon
        self.range = range
        
        
        self.timesteps = timesteps
        self.samples = samples
        self.num_epochs = num_epochs
        self.batch_size = batch_size


        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.n

        self.actor = ActorNetwork(self.obs_dim,hidden_size = 32 ,act_dim=self.act_dim).apply(
                orthogonal_init)

        self.critic = CriticNetwork(self.obs_dim,hidden_size=64).apply(orthogonal_init)

        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        
        self.trajectories = History()

        
        self.actor_losses = []
        self.critic_losses = []
        self.scores = []
        self.steps = []

        
        

    def _get_action(self, state: np.ndarray) -> float:
        """
        Returns action based on the stae of the env and value function 
        """
        state = torch.FloatTensor(state)
        action, dist = self.actor(state)

        
        value = self.critic(state)
        
        
        self.trajectories.states.append(state)
        self.trajectories.actions.append(action)
        self.trajectories.log_probs.append(dist.log_prob(action))
        self.trajectories.values.append(value)

        return list(action.detach().cpu().numpy()).pop()
            
    def take_action(self, action: float):
        """
        Based on the current policy, an environment step is taken.
        The result of this action is the next state that the environment enters 
        and the associated reward.
        """
        
        next_state, reward, done, _ = self.env.step(action)

        
        next_state = np.reshape(next_state, (1, -1)).astype(np.float64)
        reward = np.reshape(reward, (1, -1)).astype(np.float64)
        done = np.reshape(done, (1, -1))

        
           
        self.trajectories.rewards.append(torch.FloatTensor(reward))
        self.trajectories.is_done.append(torch.FloatTensor(1 - done))

        return next_state, reward, done

    def train(self):
        """
        Training of the newtorks occurs here. 
        Returns scores, actor and critic losses.
        """
        score = 0
        state = self.env.reset()
        state = np.reshape(state, (1, -1))

        for i in range(self.samples):
            self.steps.append(i)
            for _ in range(self.timesteps):
                action = self._get_action(state)
                next_state, reward, done = self.take_action(action)

                state = next_state
                score += reward[0][0]

                if done[0][0]:
                    self.scores.append(score)
                    score = 0
                    state = self.env.reset()
                    state = np.reshape(state, (1, -1))

           
            value = self.critic(torch.FloatTensor(next_state))
            self.trajectories.values.append(value)
            
            self.update_net_weights()

        actor_loss, critic_loss , scores_df  = self.plotting_graphs()
        self.env.close()
        return actor_loss, critic_loss, scores_df

    def update_net_weights(self, use_gae =True):
      """ For stability, a basline is added to the gradient update.
      Functionality includes both simple advantage and GAE. 
      PPO-Clip is implemented, which based on the sign of the advantage, 
      limits the size of  policy gradient step. Epsilon determines how far a new policy is allowed to stray from an older policy."""

      if use_gae:
          returns = compute_gen_adv_est(
            self.trajectories.rewards,
            self.trajectories.values,
            self.trajectories.is_done,
            self.gamma,
            self.lamda,
          )

      else:
          returns = self.trajectories.rewards
        
          actor_losses, critic_losses  = [], [] 
          approx_kl_divs = []

        
          states = torch.cat(self.trajectories.states).view(-1, self.obs_dim)
          actions = torch.cat(self.trajectories.actions)
          returns = torch.cat(returns).detach()
          log_probs = torch.cat(self.trajectories.log_probs).detach()
          values = torch.cat(self.trajectories.values).detach()
          advantages = returns - values[:-1]

      for state, action, return_, old_log_prob, old_value, advantage in get_trajectory(
            states=states,
            actions=actions,
            returns=returns,
            log_probs=log_probs,
            values=values,
            advantages=advantages,
            batch_size=self.batch_size,
            num_epochs=self.num_epochs,
            ):

            
            _, dist = self.actor(state)
            cur_log_prob = dist.log_prob(action)
            ratio = torch.exp(cur_log_prob - old_log_prob)
  
            approx_kl_div = torch.mean((torch.exp(ratio) - 1) - ratio)
            approx_kl_divs.append(approx_kl_div)

            if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                  break

            
            entropy = dist.entropy().mean()

            
            loss =  advantage * ratio
            clipped_loss = (
                torch.clamp(ratio, 1. - self.epsilon, 1. + self.epsilon)
                 * advantage
                )
            actor_loss = (
                -torch.mean(torch.min(loss, clipped_loss))
                - entropy * self.entropy_coef)
            
            
            cur_value = self.critic(state)
            
            critic_loss = (return_ - cur_value).pow(2).mean()

            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

           
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            end_epi += 1
            

        
      self.trajectories.clear_history()

       
      actor_loss = sum(actor_losses) / len(actor_losses)
      critic_loss = sum(critic_losses) / len(critic_losses)
      self.actor_losses.append(actor_loss)
      self.critic_losses.append(critic_loss)


    def plotting_graphs(self):
        critic_losses = pd.DataFrame({"update_iteration": self.steps, 
                      "critic_loss": self.critic_losses})
        
        actor_losses = pd.DataFrame({"update_iteration": self.steps, 
                      "critic_loss": self.actor_losses})
        
        scores = pd.DataFrame({
                      "scores": self.scores})
        
        return actor_losses ,critic_losses ,scores
