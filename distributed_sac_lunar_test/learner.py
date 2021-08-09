import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from replay_buffer import ReplayBuffer
from model import Actor, Critic
import random
import itertools
import os
import json
import collections
import time
from torch.utils.tensorboard import SummaryWriter

from utils import Decoder

import ray
import redis
import _pickle

from logger import Logger

# @ray.remote(num_gpus=0.5, num_cpus=1)
class Learner():
    def __init__(self, 
            cfg_path,
            update_delay=3,
            print_period=10,
            write_mode=True,
            save_period=1000,
            checkpoint_path=None
        ):

        self.cfg = self.cfg_read(path=cfg_path)
        self.set_cfg_parameters(print_period, save_period, write_mode, update_delay)

        self.server = redis.StrictRedis(host='localhost')
        for key in self.server.scan_iter():
            self.server.delete(key)    

        self.memory = ReplayBuffer(buffer_size=self.buffer_size, batch_size=self.batch_size, seed=0, device=self.device, server=self.server)
        self.memory.start() # start thread's activity

        self.logger = Logger(writer=self.writer, server=self.server)
        self.logger.start() # start thread's activity
            
        self.build_model()
        self.to_device()
        self.build_optimizer()

        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)
            print('######## load checkpoint completely ########')
    
    def cfg_read(self, path):
        with open(path, 'r') as f:
            cfg = json.loads(f.read(), cls=Decoder)
        return cfg

    def set_cfg_parameters(self, print_period, save_period, write_mode, update_delay):
        self.gamma = self.cfg['gamma']
        self.lr_actor = self.cfg['lr_actor']
        self.lr_critic = self.cfg['lr_critic']
        self.device = self.cfg['device']
        self.batch_size = int(self.cfg['batch_size'])
        self.tau = self.cfg['tau']                     # soft update parameter
        self.reward_scale = self.cfg['reward_scale']
        self.start_memory_len = self.cfg['start_memory_len']
        self.buffer_size = int(1e5)
        self.print_period = print_period
        self.total_step = 0
        self.episode_idx = 0
        self.update_delay = update_delay
        
        self.action_dim = 2
        self.state_dim = 8
        self.action_bound = [-1.0, 1.0]

        self.log_file = './log_distributed_test/test_log.txt'

        self.save_model_path = 'saved_models/distributed_test/'
        self.save_period = save_period
        if not os.path.exists(self.save_model_path):
            os.makedirs(self.save_model_path)

        self.write_mode = write_mode
        if self.write_mode:
            self.writer = SummaryWriter('./log_distributed_test')

    def build_model(self):

        self.actor = Actor(self.state_dim, self.action_dim, self.device, self.action_bound, hidden_dim=[256,256])
        
        self.local_critic_1 = Critic(self.state_dim, self.action_dim, self.device, hidden_dim=[256,256])
        self.local_critic_2 = Critic(self.state_dim, self.action_dim, self.device, hidden_dim=[256,256])
        self.target_critic_1 = Critic(self.state_dim, self.action_dim, self.device, hidden_dim=[256,256])
        self.target_critic_2 = Critic(self.state_dim, self.action_dim, self.device, hidden_dim=[256,256])
       
        self.H_bar = torch.tensor([-self.action_dim]).to(self.device).float() # minimum entropy
        self.log_alpha = nn.Parameter(
            torch.tensor([float(self.cfg['log_alpha'])], requires_grad=True, device=self.device).float()
        ) 
        self.alpha = self.log_alpha.exp().detach()

    def build_optimizer(self):
        # 1. actor optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)

        # 2. critic optimizer
        iterator = itertools.chain(self.local_critic_1.parameters(), self.local_critic_2.parameters())
        self.critic_optimizer = optim.Adam(iterator, lr=self.lr_critic) 

        # 3. log_alpha optimizer
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr_actor)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
    
    def my_print(self, content):
        with open(self.log_file, 'a') as writer:
            print(content)
            writer.write(content+'\n')

    def save_checkpoint(self, episode_idx):
        state = {
            'episode_idx' : episode_idx,
            'total_step' : self.total_step,

            'local_critic_1' : {k: v.cpu() for k, v in self.local_critic_1.state_dict().items()},
            'local_critic_2' : {k: v.cpu() for k, v in self.local_critic_2.state_dict().items()},            
            'critic_optimizer' : self.critic_optimizer.state_dict(),

            'target_critic_1' : {k: v.cpu() for k, v in self.target_critic_1.state_dict().items()},
            'target_critic_2' : {k: v.cpu() for k, v in self.target_critic_2.state_dict().items()},

            'actor' : {k: v.cpu() for k, v in self.actor.state_dict().items()},
            'actor_optimizer' : self.actor_optimizer.state_dict(),

            'log_alpha' : self.log_alpha.cpu(),
            'log_alpha_optimizer' : self.log_alpha_optimizer.state_dict(),
            'alpha' : self.alpha.cpu()                
        }
        torch.save(state, self.save_model_path+'lunar_sac_checkpoint_{}.tar'.format(str(episode_idx))) 
        
    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.episode_idx = checkpoint['episode_idx']
        self.total_step = checkpoint['total_step']

        self.local_critic_1.load_state_dict(checkpoint['local_critic_1'])
        self.local_critic_2.load_state_dict(checkpoint['local_critic_2'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])

        self.target_critic_1.load_state_dict(checkpoint['target_critic_1'])
        self.target_critic_2.load_state_dict(checkpoint['target_critic_2'])

        self.actor.load_state_dict(checkpoint['actor'])
        self.actor.optimizer.load_state_dict(checkpoint['actor_optimizer'])

        self.log_alpha.data = checkpoint['log_alpha']
        self.log_alpha_optimizer.load_state_dict(checkpoint['log_alpha_optimizer'])
        self.alpha.data = checkpoint['alpha']
        
    def to_device(self):        
        self.actor.to(self.device)
        self.local_critic_1.to(self.device)
        self.local_critic_1.to(self.device)
        self.target_critic_1.to(self.device)
        self.target_critic_1.to(self.device)

    def write(self, update_iteration, critic_loss, actor_loss):

        critic_loss_data = (update_iteration, critic_loss)
        self.server.rpush('critic_loss', _pickle.dumps(critic_loss_data))
        
        actor_loss_data = (update_iteration, actor_loss)
        self.server.rpush('actor_loss', _pickle.dumps(actor_loss_data))

        alphas = self.log_alpha.exp().detach().cpu().numpy()
        alphas_data = (update_iteration, alphas)
        self.server.rpush('alpha', _pickle.dumps(alphas_data))
        
    def update_SAC(self, states, actions, rewards, next_states, dones, alpha, retain_graph=True):

        # Compute targets for the Q functions
        with torch.no_grad():
            sampled_next_actions, next_log_probs = self.actor.get_action_log_prob(next_states)
            Q_target_1 = self.target_critic_1.forward(next_states, sampled_next_actions).detach()
            Q_target_2 = self.target_critic_2.forward(next_states, sampled_next_actions).detach()
            y = self.reward_scale*rewards + self.gamma * (1-dones)*(torch.min(Q_target_1, Q_target_2) - alpha*next_log_probs)

        # Update Q-functions by one step of gradient descent       
        Q_loss_1 = self.local_critic_1.cal_loss(states=states, actions=actions, td_target_values=y)
        Q_loss_2 = self.local_critic_2.cal_loss(states=states, actions=actions, td_target_values=y)
        Q_loss = Q_loss_1 + Q_loss_2
        Q_loss.backward(retain_graph=retain_graph)
        self.critic_optimizer.step()           
        critic_loss = Q_loss.item()

        # Update policy by one step of gradient ascent
        sampled_actions, log_probs = self.actor.get_action_log_prob(states)
        Q_min = torch.min(self.local_critic_1.forward(states, sampled_actions), self.local_critic_2.forward(states, sampled_actions))
        policy_loss = self.actor.cal_loss(log_probs, Q_min, alpha)
        policy_loss.backward(retain_graph=retain_graph)
        self.actor_optimizer.step()
        actor_loss = policy_loss.item()

        #### MT SAC ####
        # Adjust temperature
        loss_log_alpha = -(self.log_alpha * (log_probs.detach() + self.H_bar)).mean()  
        loss_log_alpha.backward(retain_graph=retain_graph)
        self.log_alpha_optimizer.step()
        #### MT SAC ####

        # Update target networks
        self.soft_update(self.local_critic_1, self.target_critic_1, self.tau)
        self.soft_update(self.local_critic_2, self.target_critic_2, self.tau)

        return critic_loss, actor_loss

    def optimizer_zero_grad(self):
        self.critic_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()
        self.log_alpha_optimizer.zero_grad()
    
    def update(self):
        # Randomly sample a batch of trainsitions from D
        states, actions, rewards, next_states, dones = self.memory.sample()

        alpha = self.log_alpha.exp().detach() 

        self.optimizer_zero_grad()

        #### update SAC ####
        critic_loss, actor_loss = self.update_SAC(
            states, 
            actions, 
            rewards, 
            next_states, 
            dones, 
            alpha, 
            retain_graph=False
        )
        return critic_loss, actor_loss

    def wait_until_memoryReady(self):
        while True:
            if len(self.memory) > self.start_memory_len:
                break
            time.sleep(0.1)

    def get_parameters(self):
        parameters = {
            'actor' : {k: v.cpu() for k, v in self.actor.state_dict().items()},
        }
        return parameters


    def run(self):
        # initial parameter copy
        self.server.set('update_iteration', _pickle.dumps(-1))
        self.server.set('parameters', _pickle.dumps(self.get_parameters())) # parameters for state_encoder, actor
        
        self.wait_until_memoryReady()
        self.my_print('######################### Start train #########################')

        # copy parameters to target 
        self.soft_update(self.local_critic_1, self.target_critic_1, 1.0)
        self.soft_update(self.local_critic_2, self.target_critic_2, 1.0)

        actor_loss_list = []
        critic_loss_list = []

        for update_iteration in itertools.count():
            if update_iteration % self.update_delay != 0 : continue
            critic_loss, actor_loss = self.update()

            self.server.set('update_iteration', _pickle.dumps(update_iteration))
            self.server.set('parameters', _pickle.dumps(self.get_parameters())) # parameters for state_encoder, actor

            if self.write_mode:
                self.write(update_iteration, critic_loss, actor_loss)
                actor_loss_list.append(actor_loss)
                critic_loss_list.append(critic_loss)
                if update_iteration % self.print_period == 0:
                    content = '[Learner] Update_iteration: {0:<6} \t | actor_loss : {1:5.3f} \t | critic_loss : {2:5.3f}'.format(update_iteration, actor_loss, critic_loss)
                    self.my_print(content)
                    actor_loss_list = []
                    critic_loss_list = []

            if update_iteration % self.save_period == 0:
                self.save_checkpoint(update_iteration)