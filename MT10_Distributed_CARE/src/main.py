from player import Player
from learner import Learner
import metaworld
import ray
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

cfg_path = 'cfg/MT10_Distributed_CARE_cfg.json'
MT10 = metaworld.MT10()
train_classes = MT10.train_classes
train_tasks = MT10.train_tasks

# is_train = True ############## you need to set 
is_train = False ############## you need to set 

num_cpus = 14 ############## you need to set 
num_gpus = 1 ############## you need to set 
Player = ray.remote(num_cpus=3, num_gpus=0.15)(Player) ############## you need to set 
Learner = ray.remote(num_cpus=2, num_gpus=0.3)(Learner) ############## you need to set 
ray.init(num_cpus=num_cpus, num_gpus=num_gpus)


############## you need to set ##############
# task_distributions = [[0, 2, 5], [1, 3, 7], [4, 6, 8, 9]] # 3 players
# task_distributions = [[0, 2], [5, 7], [1, 3, 4], [6, 8, 9]] # 4 players
task_distributions = [[0, 5, 8], [2, 7], [1, 3, 9], [4, 6]] # 4 players
# task_distributions = [[0, 2], [5, 7], [1, 3], [6, 8], [4, 9]] # 5 players
# task_distributions = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]] # 10 players
############## you need to set ##############

networks = []
if is_train:
    for task_idx_list in task_distributions:
        networks.append(
            Player.remote(
                train_classes, 
                train_tasks, 
                cfg_path, 
                task_idx_list,
                eval_episode_idx=40
            )
        )
    networks.append(Learner.remote(train_classes, train_tasks, cfg_path, save_period=20000))
    print('Learner added')
else:
    task_idx_list = [7] ############## you need to set 
    for _ in range(1):
        networks.append(
            Player.remote(
                train_classes,
                train_tasks, 
                cfg_path, 
                task_idx_list, 
                train_mode=False, 
                trained_model_path='saved_models/MT10_Distributed_CARE/CARE(M)/checkpoint_6300000.tar', 
                write_mode=False, 
                render_mode=True, 
                eval_episode_idx=400
            )
        )
        # checkpoint_5940000
        # checkpoint_4860000
        # checkpoint_4920000

ray.get([network.run.remote() for network in networks])
ray.shutdown()
