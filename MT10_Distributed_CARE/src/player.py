import torch
import numpy as np
from model import Actor
import random
import time

from utils import cfg_read

import redis
import _pickle

from context_encoder import contextEncoder


class Player():
    def __init__(self, 
            train_classes,
            train_tasks,
            cfg_path,
            task_idx_list,
            train_mode=True,
            trained_model_path=None,
            render_mode=False,
            write_mode=True,
            eval_episode_idx=100
        ):

        self.envs_dict = None
        self.env = None
        self.task_inital_state_dict = None

        self.train_classes = train_classes
        self.train_tasks = train_tasks
        self.task_idx_list = task_idx_list
        self.train_mode = train_mode
        self.render_mode = render_mode
        self.eval_episode_idx = eval_episode_idx

        self.cfg = cfg_read(path=cfg_path)
        self.encoder_cfg = self.cfg['encoder']
        self.set_cfg_parameters()

        self.write_mode = write_mode

        self.server = redis.StrictRedis(host='localhost')
        for key in self.server.scan_iter():
            self.server.delete(key)    
     
        self.build_model()
        self.to_device()

        if self.train_mode is False:
            assert trained_model_path is not None, \
                'Since train mode is False, trained actor path is needed.'
            self.load_model(trained_model_path)

    def set_cfg_parameters(self):
        self.update_iteration = -2

        self.device = torch.device(self.cfg['device'])
        self.reward_scale = self.cfg['reward_scale']
        self.random_step = int(self.cfg['random_step']) if self.train_mode else 0  
        self.print_period = int(self.cfg['print_period_player'])   
        self.num_tasks = int(self.cfg['num_tasks']) #### MT SAC ####        
        self.max_episode_time = self.cfg['max_episode_time'] 
        self.use_modified_care = self.cfg['use_modified_care']

    def build_model(self):
        self.context_encoder = contextEncoder(self.encoder_cfg, self.use_modified_care)
        self.actor = Actor(self.cfg['actor'], self.encoder_cfg, self.use_modified_care)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.context_encoder.load_state_dict(checkpoint['context_encoder'])
        self.actor.load_state_dict(checkpoint['actor'])
        print('########### Trained model loaded ###########')

    def to_device(self):        
        self.context_encoder.to(self.device)
        self.actor.to(self.device)

    def pull_parameters(self):
        parameters = self.server.get('parameters')
        update_iteration = self.server.get('update_iteration')

        if parameters is not None:
            if update_iteration is not None:
                update_iteration = _pickle.loads(update_iteration)
            if self.update_iteration != update_iteration:
                parameters = _pickle.loads(parameters)
                self.context_encoder.load_state_dict(parameters['context_encoder'])
                self.actor.load_state_dict(parameters['actor'])
                self.update_iteration = update_iteration
    
    def set_env(self, task_idx):
        if self.envs_dict is None:
            self.task_inital_state_dict = {}
            self.envs_dict = {}
            env_info_list = list(self.train_classes.items())
            for task_idx in self.task_idx_list:
                env_name, env_cls = env_info_list[task_idx]
                self.envs_dict[task_idx] = env_cls()
                self.task_inital_state_dict[task_idx] = [task for task in self.train_tasks
                                            if task.env_name == env_name]

        task = random.choice(self.task_inital_state_dict[task_idx])
        self.env = self.envs_dict[task_idx]
        self.env.set_task(task)
        assert self.env is not None, 'env is not set.'

    def trajectory_generator(self, task_idx):
        """Tests whether a given policy solves an environment
        Args:
            task_idx (int)

        Yields:
            (float, bool, dict): Reward, Done flag, Info dictionary
        """
        self.set_env(task_idx)
        self.env.reset()
        self.env.reset_model()

        state = self.env.reset()
        mtobs = self.state2mtobs(state, taskIdx=task_idx)

        for _ in range(self.env.max_path_length):
            with torch.no_grad():    
                mtobs_tensor = torch.tensor([mtobs]).to(self.device).float()
                z_context = self.context_encoder.forward(mtobs_tensor)
                action = self.actor.get_action(
                    mtobss=mtobs_tensor, 
                    z_context=z_context, 
                    stochastic=False
                )

            state, reward, done, info = self.env.step(action)
            mtobs = self.state2mtobs(state, taskIdx=task_idx)

            yield reward, done, info
    
    def is_success(self, task_idx, end_on_success=True):
        """Tests whether a given policy solves an environment

        Args:
            task_idx (int) : Index of the task which will be evaluated
            end_on_success (bool): Whether to stop stepping after first success
        Returns:
            Success flag (bool)
        """
        success = False
        rewards = []

        for t, (r, done, info) in enumerate(self.trajectory_generator(task_idx)):
            rewards.append(r)
            success |= bool(info['success'])
            if (success or done) and end_on_success:
                break
        return success        

    def calculate_success_rate(self, task_idx, num_eval=50):
        succes_list = [float(self.is_success(task_idx)) for _ in range(num_eval)]

        return np.sum(succes_list)/num_eval

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
    
    def run_episode_once(self, task_idx):
        '''
        Args:
            task_idx (int) : Index of the task which will be run once
        Output:
            episode_reward (float) : It is return of Episode
            delta_total_step (int)
            t (int) : consumed time horizon            
        '''
        self.set_env(task_idx)

        state = self.env.reset()
        mtobs = self.state2mtobs(state, taskIdx=task_idx)
        episode_reward = 0.
        delta_total_step = -self.task_total_step_dict[task_idx]

        for t in range(1, self.max_episode_time+1):   
            if self.task_total_step_dict[task_idx] < self.random_step:
                action = self.env.action_space.sample()
            else:         
                with torch.no_grad():    
                    mtobs_tensor = torch.tensor([mtobs]).to(self.device).float()
                    z_context = self.context_encoder.forward(mtobss=mtobs_tensor)
                    action = self.actor.get_action(
                        mtobss=mtobs_tensor, 
                        z_context=z_context, 
                        stochastic=self.train_mode
                    )

            next_state, reward, done, info = self.env.step(action)
            next_mtobs = self.state2mtobs(next_state, taskIdx=task_idx)
            episode_reward += reward

            # Ignore the "done" signal if it comes from hitting the time horizon.
            masked_done = False if t == self.max_episode_time else done

            if self.train_mode:
                sample = (
                    task_idx,
                    mtobs.copy(), 
                    action.copy(), 
                    self.reward_scale * reward, 
                    next_mtobs.copy(), 
                    masked_done
                )
                self.server.rpush('sample', _pickle.dumps(sample))

                self.pull_parameters()
            else:
                if self.render_mode:
                    self.env.render()
                    time.sleep(0.04)
                    if bool(info['success']) or t > 200:
                        time.sleep(0.5)
                        break
                    
            mtobs = next_mtobs

            self.task_total_step_dict[task_idx] += 1

            if done: 
                break

        delta_total_step += self.task_total_step_dict[task_idx]

        return episode_reward, delta_total_step, t

    def run(self):
        total_step = 0
        self.task_total_step_dict = {task_idx : 0 for task_idx in self.task_idx_list}
        episode_idx = 0
        episode_rewards_dict = {task_idx : 0. for task_idx in self.task_idx_list}
        ts_dict = {task_idx : 0. for task_idx in self.task_idx_list}

        # initial parameter copy
        self.pull_parameters()

        while True:        

            for task_idx in self.task_idx_list:
                episode_reward, delta_total_step, t = self.run_episode_once(task_idx)
                episode_rewards_dict[task_idx] += episode_reward
                total_step += delta_total_step
                ts_dict[task_idx] += t

            episode_idx += 1

            if episode_idx % self.print_period == 0:
                for task_idx in self.task_idx_list:
                    content = '[Player] Tot_step: {0:<6} \t | Episode: {1:<4} \t | Time: {2:5.2f} \t | Task: {3:<2} \t | Reward : {4:5.3f}'.format(total_step, episode_idx + 1, ts_dict[task_idx]/self.print_period, task_idx, episode_rewards_dict[task_idx]/self.print_period)
                    print(content)
                if self.write_mode:
                    for task_idx in self.task_idx_list:
                        reward_logs_data = (
                            task_idx,
                            self.task_total_step_dict[task_idx],
                            episode_rewards_dict[task_idx]/self.print_period
                        )
                        self.server.rpush('reward_logs', _pickle.dumps(reward_logs_data))
                
                episode_rewards_dict = {task_idx : 0. for task_idx in self.task_idx_list}
                ts_dict = {task_idx : 0. for task_idx in self.task_idx_list}
            
            if episode_idx % self.eval_episode_idx == 0:
                for task_idx in self.task_idx_list:
                    success_rate = self.calculate_success_rate(task_idx, num_eval=50)
                    if self.train_mode:
                        success_rate_data = (task_idx, self.task_total_step_dict[task_idx], success_rate)
                        self.server.rpush('success_rate', _pickle.dumps(success_rate_data))        
                    else:
                        print('[Task {}] Succes rate is {}'.format(task_idx, success_rate*100))
