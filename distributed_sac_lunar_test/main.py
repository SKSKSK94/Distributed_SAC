#%%
from player import Player
from learner import Learner
import torch
import gym
import numpy as np
import random
import time
import ray
import os

max_episode_num = 10000
num_cpus = 4
num_gpus = 1
is_train = True
# is_train = False


Player = ray.remote(num_cpus=1)(Player)
Learner = ray.remote(num_cpus=2, num_gpus=0)(Learner)

ray.init(num_cpus=num_cpus, num_gpus=num_gpus)

GPU_NUM = 0
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
cfg_path = '../cfg/cfg.json'

env = gym.make('LunarLanderContinuous-v2') 


#%%

networks = []
for task_idx in range(2):
    networks.append(Player.remote(env, cfg_path, task_idx))

if is_train:
    networks.append(Learner.remote(cfg_path))

ray.get([network.run.remote() for network in networks])
#%%
ray.shutdown()