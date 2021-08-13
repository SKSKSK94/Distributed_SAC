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
import metaworld

MT1 = metaworld.MT1('pick-place-v2') # Construct the benchmark, sampling tasks
train_classes = MT1.train_classes
train_tasks = MT1.train_tasks

num_cpus = 6
num_gpus = 1
# is_train = True
is_train = False

Player = ray.remote(num_cpus=1, num_gpus=0.02)(Player)
Learner = ray.remote(num_cpus=1, num_gpus=0.6)(Learner)

ray.init(num_cpus=num_cpus, num_gpus=num_gpus)

GPU_NUM = 0
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
cfg_path = 'cfg/MT1_Distributed_SAC_cfg.json'

networks = []
if is_train:
    for task_idx in range(4):
        networks.append(
            Player.remote(
                train_classes, 
                train_tasks, 
                cfg_path, 
                task_idx, 
                train_mode=True, 
                eval_episode_idx=200
            )
        )
    networks.append(Learner.remote(cfg_path))
    print('Learner added')
else:
    for task_idx in range(1):
        networks.append(
            Player.remote(
                train_classes,
                train_tasks, 
                cfg_path, 
                task_idx, 
                train_mode=False, 
                trained_actor_path='saved_models/MT1_Distributed_VSAC/checkpoint_700000.tar', 
                write_mode=False, 
                render_mode=True, 
                eval_episode_idx=20
            )
        )

ray.get([network.run.remote() for network in networks])
ray.shutdown()