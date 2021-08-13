from player import Player
from learner import Learner
import torch
import gym
import numpy as np
import random
import time
import metaworld
import ray
import os

num_cpus = 14
num_gpus = 1

# is_train = True
is_train = False

Player = ray.remote(num_cpus=3, num_gpus=0.15)(Player)
Learner = ray.remote(num_cpus=2, num_gpus=0.3)(Learner)

ray.init(num_cpus=num_cpus, num_gpus=num_gpus)

GPU_NUM = 0
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
cfg_path = 'cfg/MT1_Distributed_CARE_cfg.json'

MT1 = metaworld.MT1('pick-place-v2') # Construct the benchmark, sampling tasks
train_classes = MT1.train_classes
train_tasks = MT1.train_tasks

task_idx_list = [0]

networks = []
if is_train:
    for player_id in range(4):
        networks.append(
            Player.remote(
                player_id,
                train_classes,
                train_tasks, 
                cfg_path, 
                task_idx_list
            )
        )
    networks.append(Learner.remote(train_classes, train_tasks, cfg_path))
    print('Learner added')
else:
    for player_id in range(1):
        networks.append(
            Player.remote(
                player_id,
                train_classes,
                train_tasks, 
                cfg_path, 
                task_idx_list, 
                train_mode=False, 
                trained_model_path='saved_models/MT1_Distributed_CARE/checkpoint_750000.tar', 
                write_mode=False, 
                render_mode=True, 
                eval_episode_idx=40
            )
        )

ray.get([network.run.remote() for network in networks])
ray.shutdown()

