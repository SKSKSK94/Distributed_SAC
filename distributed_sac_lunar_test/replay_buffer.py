import numpy as np
import random
import copy
from collections import namedtuple, deque

import torch

import threading
import redis
import _pickle
import time


class ReplayBuffer(threading.Thread):
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, 
        buffer_size, 
        batch_size, 
        seed, 
        device,
        server=redis.StrictRedis(host='localhost')
    ):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        super(ReplayBuffer, self).__init__()

        self.setDaemon(True)
        self.lock = threading.Lock()

        self.server = server
        self.server.delete('sample')    

        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = device

    def run(self):
        while True:
            pipe = self.server.pipeline()
            pipe.lrange('sample', 0, -1) # get values of keys(sample) from start=0 to stop=-1(end)
            pipe.ltrim('sample', -1, 0) # remove values of keys(sample) except start=-1(end) to stop=0(end)
            
            # the "EXECUTE" call sends all buffered commands to the server, returning
            # a list of responses, one for each command.
            datas, _ = pipe.execute() 

            if datas is not None:
                for data in datas:
                    data = _pickle.loads(data)
                    state, action, reward, next_state, done = data
                    e = self.experience(state, action, reward, next_state, done)
                    with self.lock:
                        self.memory.append(e)

            time.sleep(0.01)

    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)