import torch
import threading
import redis
import _pickle
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class Logger(threading.Thread):
    def __init__(self,
        writer : SummaryWriter,
        server=redis.StrictRedis(host='localhost')
    ):
        super(Logger, self).__init__()

        self.writer = writer

        self.setDaemon(True)
        self.lock = threading.Lock()

        self.server = server
        self.server.delete('reward_logs')   
        self.server.delete('critic_loss') 
        self.server.delete('actor_loss') 
        self.server.delete('alpha')  
    
    def run(self):
        while True:
            self.write_reward()      
            self.write_loss()      

            time.sleep(0.01)

    def write_reward(self):
        pipe = self.server.pipeline()
        pipe.lrange('reward_logs', 0, -1) # get values of keys(reward_logs) from start=0 to stop=-1(end)
        pipe.ltrim('reward_logs', -1, 0) # remove values of keys(reward_logs) except start=-1(end) to stop=0(end)
        
        # the "EXECUTE" call sends all buffered commands to the server, returning
        # a list of responses, one for each command.
        datas, _ = pipe.execute() 

        if datas is not None:
            for data in datas:
                data = _pickle.loads(data)
                idx, total_step, reward = data
                reward_dict = {'learner_{}'.format(idx) : reward}

                with self.lock:
                    self.writer.add_scalars('Reward/reward', reward_dict, total_step)

    def write_loss(self):
        pipe = self.server.pipeline()

        pipe.lrange('critic_loss', 0, -1) # get values of keys(critic_loss) from start=0 to stop=-1(end)
        pipe.ltrim('critic_loss', -1, 0) # remove values of keys(critic_loss) except start=-1(end) to stop=0(end)
        
        pipe.lrange('actor_loss', 0, -1) # get values of keys(actor_loss) from start=0 to stop=-1(end)
        pipe.ltrim('actor_loss', -1, 0) # remove values of keys(actor_loss) except start=-1(end) to stop=0(end)
        
        pipe.lrange('alpha', 0, -1) # get values of keys(alpha) from start=0 to stop=-1(end)
        pipe.ltrim('alpha', -1, 0) # remove values of keys(alpha) except start=-1(end) to stop=0(end)
        
        # the "EXECUTE" call sends all buffered commands to the server, returning
        # a list of responses, one for each command.
        critic_datas, _, actor_datas, _, alpha_datas, _ = pipe.execute() 

        if critic_datas is not None:
            for data in critic_datas:
                data = _pickle.loads(data)
                update_iteration, critic_loss = data
                
                with self.lock:
                    self.writer.add_scalar('Loss/critic_loss', critic_loss, update_iteration)        

        if actor_datas is not None:
            for data in actor_datas:
                data = _pickle.loads(data)
                update_iteration, actor_loss = data
                
                with self.lock:
                    self.writer.add_scalar('Loss/actor_loss', actor_loss, update_iteration)
        
        if alpha_datas is not None:
            for data in alpha_datas:
                data = _pickle.loads(data)
                update_iteration, alphas = data
                alpha_dict = {'task'+str(idx) : alpha_ for idx, alpha_ in enumerate(alphas)}
                
                with self.lock:
                    self.writer.add_scalars('Alpha/alpha', alpha_dict, update_iteration) 