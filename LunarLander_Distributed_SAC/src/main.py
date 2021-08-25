from player import Player
from learner import Learner
import gym
import ray
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

num_cpus = 4
num_gpus = 1

# is_train = True
is_train = False

Player = ray.remote(num_cpus=1, num_gpus=0.1)(Player)
Learner = ray.remote(num_cpus=2, num_gpus=0.3)(Learner)

ray.init(num_cpus=num_cpus, num_gpus=num_gpus)

cfg_path = 'cfg/LunarLanderContinuous-v2_Distributed_SAC_cfg.json'

env = gym.make('LunarLanderContinuous-v2') 

networks = []
if is_train:    
    for task_idx in range(2):
        networks.append(Player.remote(env, cfg_path, task_idx))
    networks.append(Learner.remote(cfg_path))
    print('Learner added')
else:    
    for task_idx in range(1):
        networks.append(
            Player.remote(
                env, 
                cfg_path, 
                task_idx, 
                train_mode=False,
                trained_actor_path='saved_models/LunarLander_Distributed_SAC/checkpoint_165000.tar',
                render_mode=True,
                write_mode=False
            )
        )

ray.get([network.run.remote() for network in networks])
ray.shutdown()
