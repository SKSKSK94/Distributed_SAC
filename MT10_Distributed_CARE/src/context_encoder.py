import torch
import torch.nn as nn
import torch.optim as optim
import json
from utils import weights_init, build_mlp, cfg_read

class contextEncoder(nn.Module):
    '''
    Params
        encoder_cfg : configuration for state encoder
        use_modified_care (bool) :
            if True, then use modified version of CARE which is
            to use weighted loss and change position of mlp as below    

            Roberta embedding 

            or

            if False, then use original version of CARE as below

            Roberta embedding -> embedding header -> mlp
    '''
    def __init__(self, encoder_cfg, use_modified_care):
        super(contextEncoder, self).__init__()
        self.hidden_dims = encoder_cfg['hidden_dims_contextEnc']
        self.embedding_dim = encoder_cfg['embedding_dim_contextEnc']
        self.output_dim = encoder_cfg['output_dim_contextEnc']
        self.use_modified_care = use_modified_care

        # Read json files
        self.taskName2pretrainedEmbbedding = cfg_read(
            path=encoder_cfg['pretrained_embedding_json_path']
        )
        self.task_name_list = cfg_read(
            path=encoder_cfg['task_name_json_path']
        )
        self.num_tasks = len(self.task_name_list)

        # Figure out pretrained embedding dim
        keys = list(self.taskName2pretrainedEmbbedding.keys())
        self.input_dim = len(self.taskName2pretrainedEmbbedding[keys[0]])

        # 1. Roberta embedding
        taskIdx2pretrainedEmbbedding = torch.tensor(
            [self.taskName2pretrainedEmbbedding[task_name] for task_name in self.task_name_list]
        )
        pretrained_embedding_dim = taskIdx2pretrainedEmbbedding.shape[1]
        taskIdx2pretrainedEmbbedding = nn.Embedding.from_pretrained(
            embeddings=taskIdx2pretrainedEmbbedding,
            freeze=True
        )   

        if use_modified_care:
            # This is modified CARE
            self.embedding = nn.Sequential(  
                taskIdx2pretrainedEmbbedding
            ) 
        else: 
            # This is original CARE
            # 2. embedding header
            embedding_header = nn.Sequential(
                nn.Linear(
                    in_features=pretrained_embedding_dim, out_features=2*self.embedding_dim
                ),
                nn.ReLU(),
                nn.Linear(in_features=2*self.embedding_dim, out_features=self.embedding_dim),
                nn.ReLU()
            )
            embedding_header.apply(weights_init)

            # combine 1, 2
            self.embedding = nn.Sequential(  
                taskIdx2pretrainedEmbbedding,
                nn.ReLU(),
                embedding_header
            )
            
            # 3. mlp
            self.mlp = build_mlp(
                input_dim=self.embedding_dim,
                output_dim=self.output_dim,
                hidden_dims=self.hidden_dims
            )
            self.mlp.apply(weights_init)

    def mtobss2states_taskIndices(self, mtobss : torch.Tensor) -> (torch.Tensor, torch.Tensor):
        '''
        input :
            mtobss = (batch_size, state_dim+num_tasks)
        output :
            states = (batch_size, state_dim)
            taskIndices : (batch_size, )
        '''
        one_hots = mtobss[:, -self.num_tasks:] # one_hots = (batch_size, num_tasks)
        assert one_hots.shape[1] == self.num_tasks, 'The number of tasks does not match self.num_tasks'

        states = mtobss[:, :-self.num_tasks]
        taskIndices = torch.argmax(one_hots, dim=1)

        return states, taskIndices

    def forward(self, mtobss):      
        '''
        input :
            mtobss = (batch_size, mtobs_dim) = (batch_size, state_dim + num_tasks)
        output :
            z_context = (batch_size, z_context_dim)
            here z_context_dim is hidden_dim[-1] where default is 50  
        '''
        _, taskIndices = self.mtobss2states_taskIndices(mtobss)

        if self.use_modified_care:
            z_context = self.embedding(taskIndices)
        else:
            z_context = self.mlp(self.embedding(taskIndices))

        return z_context

