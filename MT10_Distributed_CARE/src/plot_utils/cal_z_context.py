from distributed_CARE_src.learner import Learner
import torch
import numpy as np
import metaworld
import torch.nn.functional as F

MT10 = metaworld.MT10()
train_classes = MT10.train_classes
train_tasks = MT10.train_tasks
GPU_NUM = 0
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')

cfg_path = 'cfg/distributed_CARE_cfg.json'

learner = Learner(
    train_classes=train_classes,
    train_tasks=train_tasks,
    cfg_path=cfg_path,
    device=device,
    write_mode=False,
    checkpoint_path='saved_models/distributed_CARE/2021-08-21 18:09:48/checkpoint_5280000.tar'
)

mtobss = torch.zeros((10, 49)).to(device)
for task_idx in range(10):
    mtobss[task_idx, 39+task_idx] = 1.0
z_context = learner.context_encoder.forward(mtobss)
print(z_context)

alphas = learner.actor.state_encoder.trunk.forward(z_context)
alphas = F.softmax(alphas, dim=1).detach().cpu()
print(alphas)

z_context_after = learner.actor.state_encoder.mlp_context.forward(z_context)
print(z_context_after.detach().cpu())