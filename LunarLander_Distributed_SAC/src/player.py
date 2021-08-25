import torch
import numpy as np
from model import Actor
import json
import time

from utils import Decoder

import redis
import _pickle


class Player():
    def __init__(self, 
            env,
            cfg_path,
            player_idx,
            train_mode=True,
            trained_actor_path=None,
            render_mode=False,
            print_period=1,
            write_mode=True
        ):

        self.env = env

        self.player_idx = player_idx
        self.train_mode = train_mode
        self.render_mode = render_mode
        self.cfg = self.cfg_read(path=cfg_path)
        self.set_cfg_parameters(print_period)

        self.write_mode = write_mode

        self.server = redis.StrictRedis(host='localhost')
        for key in self.server.scan_iter():
            self.server.delete(key)    
     
        self.build_model()
        self.to_device()

        if self.train_mode is False:
            assert trained_actor_path is not None, 'Since train mode is False, trained actor path is needed.'
            self.load_model(trained_actor_path)

    def cfg_read(self, path):
        with open(path, 'r') as f:
            cfg = json.loads(f.read(), cls=Decoder)
        return cfg

    def set_cfg_parameters(self, print_period):
        self.update_iteration = -2

        self.device = torch.device(self.cfg['device'])
        self.reward_scale = self.cfg['reward_scale']
        self.random_step = 500 if self.train_mode else 0
        self.print_period = print_period
        
        self.action_dim = 2
        self.state_dim = 8
        self.action_bound = [-1.0, 1.0]
        self.max_episode_time = 1000 # maximum episode time for the given environment

    def build_model(self):
        self.actor = Actor(self.state_dim, self.action_dim, self.device, self.action_bound, hidden_dim=[256, 256])

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        print('########### Trained actor loaded ###########')

    def to_device(self):        
        self.actor.to(self.device)

    def pull_parameters(self):
        parameters = self.server.get('parameters')
        update_iteration = self.server.get('update_iteration')

        if parameters is not None:
            if update_iteration is not None:
                update_iteration = _pickle.loads(update_iteration)
            if self.update_iteration != update_iteration:
                parameters = _pickle.loads(parameters)
                self.actor.load_state_dict(parameters['actor'])
                self.update_iteration = update_iteration
    
    def run(self):
        total_step = 0
        episode_idx = 0
        episode_rewards = []
        ts = []

        # initial parameter copy
        self.pull_parameters()

        while True:        
            state = self.env.reset()
            episode_reward = 0.

            for t in range(1, self.max_episode_time+1):   
                if total_step < self.random_step:
                    action = self.env.action_space.sample()
                else:         
                    with torch.no_grad():    
                        action = self.actor.get_action(torch.tensor([state]).to(self.device).float())
                    action = action.squeeze(dim=0).detach().cpu().numpy()

                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward

                # Ignore the "done" signal if it comes from hitting the time horizon.
                masked_done = False if t == self.max_episode_time else done

                if self.train_mode:
                    sample = (
                        state.copy(), 
                        action.copy(), 
                        self.reward_scale * reward, 
                        next_state.copy(), 
                        masked_done
                    )
                    self.server.rpush('sample', _pickle.dumps(sample))

                    self.pull_parameters()
                else:
                    if self.render_mode:
                        self.env.render()
                        time.sleep(0.01)

                state = next_state

                total_step += 1

                if done: 
                    break

            episode_idx += 1
            episode_rewards.append(episode_reward)
            ts.append(t)

            if episode_idx % self.print_period == 0:
                content = '[Player] Tot_step: {0:<6} \t | Episode: {1:<4} \t | Time: {2:5.2f} \t | Reward : {3:5.3f}'.format(
                    total_step,
                    episode_idx + 1,
                    np.mean(ts),
                    np.mean(episode_rewards)
                )
                print(content)
                if self.write_mode:
                    reward_logs_data = (self.player_idx, total_step, np.mean(episode_rewards))
                    self.server.rpush('reward_logs', _pickle.dumps(reward_logs_data))
                episode_rewards = []
                ts = []