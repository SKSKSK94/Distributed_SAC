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

from utils import Decoder, cfg_read

import ray
import redis
import _pickle
import copy

from logger import Logger

from context_encoder import contextEncoder

class Learner():
    def __init__(self, 
            train_classes,
            train_tasks,
            cfg_path,
            write_mode=True,
            save_period=50000,
            checkpoint_path=None
        ):

        self.train_classes = train_classes
        self.train_tasks = train_tasks

        self.cfg = cfg_read(path=cfg_path)
        self.actor_cfg = self.cfg['actor']
        self.critic_cfg = self.cfg['critic']
        self.encoder_cfg = self.cfg['encoder']
        self.set_cfg_parameters(save_period, write_mode)

        self.server = redis.StrictRedis(host='localhost')
        for key in self.server.scan_iter():
            self.server.delete(key)    
            
        self.build_model()
        self.to_device()
        self.build_optimizer()

        self.memory = ReplayBuffer(
            buffer_size=self.buffer_size, 
            batch_size=self.batch_size, 
            seed=0, 
            device=self.device, 
            server=self.server
        )
        self.memory.start() # start thread's activity

        self.logger = Logger(writer=self.writer, cfg_path=cfg_path, server=self.server)
        self.logger.start() # start thread's activity

        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)
            print('######## load checkpoint completely ########')
    
    def set_cfg_parameters(self, save_period, write_mode):
        self.gamma = self.cfg['gamma']
        self.lr_actor = self.actor_cfg['lr_actor']
        self.lr_critic = self.critic_cfg['lr_critic']
        self.device = self.cfg['device']
        self.batch_size = int(self.cfg['batch_size'])
        self.tau = self.cfg['tau']                     # soft update parameter
        self.reward_scale = self.cfg['reward_scale']
        self.random_step = int(self.cfg['random_step'])        
        self.start_memory_len = self.cfg['start_memory_len']
        self.buffer_size = int(self.cfg['buffer_size'])
        self.num_tasks = int(self.cfg['num_tasks'])         #### MT SAC ####
        self.update_delay = int(self.cfg['update_delay'])
        self.print_period = int(self.cfg['print_period_learner']) * self.update_delay
        self.total_step = 0
        self.episode_idx = 0
        self.update_iteration = 0
      
        self.max_episode_time = int(self.cfg['max_episode_time']) # maximum episode time for the given environment

        self.log_file = './log_MT1_distributed_CARE/MT1_distributed_CARE_log.txt'

        self.save_model_path = 'saved_models/MT1_distributed_CARE/'
        self.save_period = save_period
        if not os.path.exists(self.save_model_path):
            os.makedirs(self.save_model_path)

        self.write_mode = write_mode
        if self.write_mode:
            self.writer = SummaryWriter('./log_MT1_distributed_CARE')

    def build_model(self):
        self.context_encoder = contextEncoder(self.encoder_cfg)

        self.actor = Actor(self.actor_cfg, self.encoder_cfg)
        
        self.local_critic = Critic(self.critic_cfg, self.encoder_cfg)
        self.target_critic = Critic(self.critic_cfg, self.encoder_cfg)
       
        self.H_bar = torch.tensor([-self.actor.action_dim]).to(self.device).float() # minimum entropy
        self.log_alpha = nn.Parameter(
            torch.tensor([float(self.cfg['log_alpha'])] * self.num_tasks, requires_grad=True, device=self.device).float()
        ) 
        self.alpha = self.log_alpha.exp().detach()

        # tie encoders between actor and critic
        self.soft_update(self.local_critic.state_encoder, self.actor.state_encoder, tau=1.0)

    

    def build_optimizer(self):
        # 1. context encoder optimizer
        self.context_encoder_optimizer = optim.Adam(self.context_encoder.parameters(), lr=self.encoder_cfg['lr_contextEnc'])
        
        # 2. actor optimizer
        self.actor_optimizer = optim.Adam(self.actor.mu_log_std_layer.parameters(), lr=self.lr_actor)

        # 3. critic optimizer
        self.critic_optimizer = optim.Adam(self.local_critic.parameters(), lr=self.lr_critic) 

        # 4. log_alpha optimizer
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

    def save_checkpoint(self, update_iteration):
        state = {
            'update_iteration' : update_iteration,
            'total_step' : self.total_step,

            'context_encoder' : {k: v.cpu() for k, v in self.context_encoder.state_dict().items()},
            'context_encoder_optimizer' : self.context_encoder_optimizer.state_dict(),

            'local_critic' : {k: v.cpu() for k, v in self.local_critic.state_dict().items()},          
            'critic_optimizer' : self.critic_optimizer.state_dict(),

            'target_critic' : {k: v.cpu() for k, v in self.target_critic.state_dict().items()},

            'actor' : {k: v.cpu() for k, v in self.actor.state_dict().items()},
            'actor_optimizer' : self.actor_optimizer.state_dict(),

            'log_alpha' : self.log_alpha.cpu(),
            'log_alpha_optimizer' : self.log_alpha_optimizer.state_dict(),
            'alpha' : self.alpha.cpu()                
        }
        torch.save(state, self.save_model_path+'checkpoint_{}.tar'.format(str(update_iteration))) 
        
    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.update_iteration = checkpoint['update_iteration']
        self.total_step = checkpoint['total_step']

        self.context_encoder.load_state_dict(checkpoint['context_encoder'])
        self.context_encoder_optimizer.load_state_dict(checkpoint['context_encoder_optimizer'])

        self.local_critic.load_state_dict(checkpoint['local_critic'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.target_critic.load_state_dict(checkpoint['target_critic'])

        self.actor.load_state_dict(checkpoint['actor'])
        self.actor.optimizer.load_state_dict(checkpoint['actor_optimizer'])

        self.log_alpha.data = checkpoint['log_alpha']
        self.log_alpha_optimizer.load_state_dict(checkpoint['log_alpha_optimizer'])
        self.alpha.data = checkpoint['alpha']
        
    def to_device(self):        
        self.context_encoder.to(self.device)
        self.actor.to(self.device)
        self.local_critic.to(self.device)
        self.target_critic.to(self.device)

    def push_log(self, update_iteration, critic_loss, actor_loss, entropy):
        
        critic_loss_data = (update_iteration, critic_loss)
        self.server.rpush('critic_loss', _pickle.dumps(critic_loss_data))
        
        actor_loss_data = (update_iteration, actor_loss)
        self.server.rpush('actor_loss', _pickle.dumps(actor_loss_data))

        entropy_data = (update_iteration, entropy)
        self.server.rpush('entropy', _pickle.dumps(entropy_data))

        alphas = self.log_alpha.exp().detach().cpu().numpy()
        alphas_data = (update_iteration, alphas)
        self.server.rpush('alpha', _pickle.dumps(alphas_data))
        
            
    #### MT SAC ####
    def get_log_alpha(self, mtobss):
        '''
        input :
            mtobss = (batch_size, mtobs_dim) where mtobs_dim = state_dim + num_tasks
            Task info should be inserted to mtobss by the form of one-hot vector
        output :
            log_alpha = (batch_size, 1) which is the log_alpha corresponding to the task
        '''
        assert mtobss.shape[-1] == self.actor.mtobs_dim, 'Input should be mtobss whose shape is (batch_size, mtobs_dim).'
        
        one_hots = mtobss[:, -self.num_tasks:] # one_hots = (batch_size, num_tasks)
        assert one_hots.shape[1] == self.num_tasks, 'The number of tasks does not match self.num_tasks'

        log_alpha = self.log_alpha # self.log_alpha = (num_tasks, )
        log_alpha = torch.matmul(one_hots, log_alpha.unsqueeze(dim=0).t()) # (batch_size, num_tasks) * (num_tasks, 1) -> (batch_size, 1)

        # task_indices = torch.argmax(one_hots, dim=1)
        # log_alpha = self.log_alpha[task_indices]

        return log_alpha
    #### MT SAC ####

    def taskIdx2oneHot(self, taskIdx):
        one_hot = np.zeros((self.num_tasks,))
        one_hot[taskIdx] = 1.
        return one_hot

    def state2mtobs(self, state, taskIdx):
        '''
        input :
            state = (state_dim)
            taskIdx : int
        output :
            mtobs = (state_dim+num_tasks)
        '''
        one_hot = self.taskIdx2oneHot(taskIdx)
        mtobs = np.concatenate((state, one_hot), axis=0)
        return mtobs

    def update_SAC(self, mtobss, actions, rewards, next_mtobss, dones, alpha, retain_graph=True):

        z_context = self.context_encoder.forward(mtobss)     

        # Compute targets for the Q functions
        with torch.no_grad():
            sampled_next_actions, next_log_probs, _, = self.actor.get_action_log_prob_log_std(
                mtobss=next_mtobss,
                z_context=z_context
            )
            Q_target_1, Q_target_2 = self.target_critic.forward(
                mtobss=next_mtobss,
                z_context=z_context,
                action=sampled_next_actions
            )
            y = self.reward_scale*rewards + self.gamma * (1-dones)*(
                torch.min(Q_target_1, Q_target_2).detach() - alpha*next_log_probs
            )

        # Update Q-functions by one step of gradient descent   
        Q_loss_1, Q_loss_2 = self.local_critic.cal_loss(
            mtobss=mtobss,
            z_context=z_context, 
            action=actions, 
            td_target_values=y
        )
        Q_loss = Q_loss_1 + Q_loss_2
        Q_loss.backward(retain_graph=retain_graph)
        self.critic_optimizer.step()           
        critic_loss = Q_loss.item()

        # Update policy by one step of gradient ascent
        # detach mixture encoder and context encoder, so we don't update it with the actor loss
        sampled_actions, log_probs, log_stds = self.actor.get_action_log_prob_log_std(
            mtobss=mtobss,
            z_context=z_context.detach(),
            detach_z_encs=True
        )
        Q1, Q2 = self.local_critic.forward(
            mtobss=mtobss,
            z_context=z_context.detach(), 
            action=sampled_actions, 
            detach_z_encs=True
        )
        Q_min = torch.min(Q1, Q2)
        policy_loss = self.actor.cal_loss(log_probs, Q_min, alpha)
        policy_loss.backward(retain_graph=False)
        self.actor_optimizer.step()
        actor_loss = policy_loss.item()
        # log_stds = (batch_size, action_dim)
        entropy = (0.5 * log_stds.shape[1] * (1.0 + np.log(2 * np.pi)) + log_stds.sum(dim=-1)).mean()
        entropy = entropy.item()

        #### MT SAC ####
        # Adjust temperature
        log_alpha = self.get_log_alpha(mtobss) # log_alpha = (batch_size, 1)
        loss_log_alpha = -(log_alpha * (log_probs.detach() + self.H_bar)).mean()  
        loss_log_alpha.backward(retain_graph=False)
        self.log_alpha_optimizer.step()
        #### MT SAC ####

        # Update target networks
        self.soft_update(self.local_critic.Q_function_1, self.target_critic.Q_function_1, self.tau)
        self.soft_update(self.local_critic.Q_function_2, self.target_critic.Q_function_2, self.tau)
        self.soft_update(self.local_critic.state_encoder, self.target_critic.state_encoder, 0.05)

        return critic_loss, actor_loss, entropy

    def optimizer_zero_grad(self):
        self.critic_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()
        self.log_alpha_optimizer.zero_grad()
        self.context_encoder.zero_grad()
    
    def update(self):
        # Randomly sample a batch of trainsitions from D
        mtobss, actions, rewards, next_mtobss, dones = self.memory.sample()

        #### MT SAC ####
        alpha = self.get_log_alpha(mtobss).exp().detach() # alpha corresponding to the task
        #### MT SAC ####

        self.optimizer_zero_grad()

        #### update SAC ####
        critic_loss, actor_loss, entropy = self.update_SAC(
            mtobss, 
            actions, 
            rewards, 
            next_mtobss, 
            dones, 
            alpha, 
            retain_graph=True
        )

        #### update state encoder ####
        self.context_encoder_optimizer.step()

        #### tie encoders between actor and critic ####
        self.soft_update(self.local_critic.state_encoder, self.actor.state_encoder, tau=1.0)

        return critic_loss, actor_loss, entropy

    def wait_until_memoryReady(self):
        while True:
            if len(self.memory) > self.start_memory_len:
                break
            time.sleep(0.1)

    def get_parameters(self):
        parameters = {
            'context_encoder' : {k: v.cpu() for k, v in self.context_encoder.state_dict().items()},
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
        self.soft_update(self.local_critic, self.target_critic, 1.0)

        actor_loss_list = []
        critic_loss_list = []
        entropy_list = []

        for update_iteration in itertools.count():            
            update_iteration = update_iteration + self.update_iteration if update_iteration == 0 else update_iteration

            if update_iteration % self.update_delay != 0 : continue

            critic_loss, actor_loss, entropy = self.update()

            self.server.set('update_iteration', _pickle.dumps(update_iteration))
            self.server.set('parameters', _pickle.dumps(self.get_parameters())) # parameters for state_encoder, actor

            if self.write_mode:
                actor_loss_list.append(actor_loss)
                critic_loss_list.append(critic_loss)
                entropy_list.append(entropy)                
                self.push_log(
                    copy.deepcopy(update_iteration), 
                    copy.deepcopy(critic_loss), 
                    copy.deepcopy(actor_loss), 
                    copy.deepcopy(entropy)
                )
                if update_iteration % self.print_period == 0:
                    content = '[Learner] Update_iteration: {0:<6} \t | actor_loss : {1:5.3f} \t | critic_loss : {2:5.3f}'.format(update_iteration, np.mean(actor_loss_list), np.mean(critic_loss_list))
                    self.my_print(content)
                    actor_loss_list = []
                    critic_loss_list = []
                    entropy_list = []

            if update_iteration % self.save_period == 0:
                self.save_checkpoint(update_iteration)