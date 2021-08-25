import numpy as np
import random
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
        num_tasks,
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

        self.buffer_size_per_task = buffer_size // num_tasks
        self.memories = [deque(maxlen=self.buffer_size_per_task) for _ in range(num_tasks)]

        self.batch_size = batch_size
        self.batch_size_per_task = batch_size // num_tasks

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
                    task_idx, state, action, reward, next_state, done = data
                    e = self.experience(state, action, reward, next_state, done)
                    with self.lock:
                        self.memories[task_idx].append(e)

            time.sleep(0.01)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        states_batch = []
        actions_batch = []
        rewards_batch = []
        next_states_batch = []
        dones_batch = []
        for memory in self.memories:            
            experiences = random.sample(memory, k=self.batch_size_per_task)

            for e in experiences:
                if e is not None:
                    states_batch.append(e.state)
                    actions_batch.append(e.action)
                    rewards_batch.append(e.reward)
                    next_states_batch.append(e.next_state)
                    dones_batch.append(e.done)
        
        shuffled_idx = np.arange(len(states_batch))
        np.random.shuffle(shuffled_idx)

        states_batch = torch.tensor(np.vstack(states_batch)).float().to(self.device)
        actions_batch = torch.tensor(np.vstack(actions_batch)).float().to(self.device)
        rewards_batch = torch.tensor(np.vstack(rewards_batch)).float().to(self.device)
        next_states_batch = torch.tensor(np.vstack(next_states_batch)).float().to(self.device)
        dones_batch = torch.tensor(np.vstack(dones_batch)).float().to(self.device)

        states_batch = states_batch[shuffled_idx]
        actions_batch = actions_batch[shuffled_idx]
        rewards_batch = rewards_batch[shuffled_idx]
        next_states_batch = next_states_batch[shuffled_idx]
        dones_batch = dones_batch[shuffled_idx]

        return (states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch)

    def __len__(self):
        """Return the current size of internal memory."""
        min_len = np.inf
        for memory in self.memories:
            min_len = min(min_len, len(memory))
        return min_len