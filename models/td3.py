import os
import math
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
#         print(self.l1.weight)
#         print(1. / math.sqrt(self.l1.weight.size(1)))
#         torch.nn.init.xavier_uniform_(self.l1.weight)
#         torch.nn.init.zeros_(self.l1.bias)
        
        self.l2 = nn.Linear(400, 300)
#         torch.nn.init.xavier_uniform_(self.l2.weight)
#         torch.nn.init.zeros_(self.l2.bias)
        
        self.l3 = nn.Linear(300, action_dim)
#         torch.nn.init.xavier_uniform_(self.l3.weight)
#         torch.nn.init.zeros_(self.l3.bias)
        
        self.max_action = max_action


    def forward(self, x):
#         print(f'actor input: {x}')
#         print(f'l1 weights: {self.l1.weight}')
#         print(f'l1 bias: {self.l1.bias}')
#         print(f'l1 output: {self.l1(x)}')
        x = F.relu(self.l1(x))
#         print(f'l1 activations: {x}')
        x = F.relu(self.l2(x))
#         print(f'l2 activations: {x}')
#         print(f'l3 activations: {torch.tanh(self.l3(x))}')
        x = self.max_action * torch.tanh(self.l3(x)) 
#         print(f'max_action scaling factor: {self.max_action}')
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)


    def forward(self, x, u):
        xu = torch.cat([x, u], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)

        x2 = F.relu(self.l4(xu))
        x2 = F.relu(self.l5(x2))
        x2 = self.l6(x2)
        return x1, x2


    def Q1(self, x, u):
        xu = torch.cat([x, u], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)
        return x1 


class TD3(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())        

        self.max_action = max_action


    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()


    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):

        for it in tqdm(range(iterations), desc='Train model', unit='batch'):

            # Sample replay buffer 
            x, y, u, r, d = replay_buffer.sample(batch_size)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(1 - d.reshape(-1,1)).to(device)
            reward = torch.FloatTensor(r.reshape(-1,1)).to(device)

            # Select action according to policy and add clipped noise 
            noise = torch.FloatTensor(u).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            next_action = next_action.clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (done * discount * target_Q).detach()

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) 

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Delayed policy updates
            if it % policy_freq == 0:

                # Compute actor loss
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
                
                # Optimize the actor 
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


    def save(self, directory, filename):
        actor_path = '%s/%s_actor.pth' % (directory, filename)
        critic_path = '%s/%s_critic.pth' % (directory, filename)
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)


    def load(self, directory, filename):
        actor_path = '%s/%s_actor.pth' % (directory, filename)
        critic_path = '%s/%s_critic.pth' % (directory, filename)
        if not os.path.exists(actor_path) or not os.path.exists(critic_path):
            raise RuntimeError(f'File not found: one or both of `{actor_path}`, `{critic_path}`.')
        self.actor.load_state_dict(torch.load(actor_path))
        self.critic.load_state_dict(torch.load(critic_path))
        
        