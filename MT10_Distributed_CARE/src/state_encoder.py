import torch
import torch.nn as nn

from utils import weights_init, build_mlp


class stateEncoder(nn.Module):
    '''
    state encoder consists of
        1. Mixture of k Encoders(Here i use feedforward encoder as base encoder)
        2. Attention over encoder represenations
    it takes as inputs
        1. mtobs which includes state and task_one_hot
        2. z_context which is encoded by context encoder using metadata 
    and return encoded_state which is the input of policy algorithm(actor and critic)
    '''
    def __init__(self, encoder_cfg, use_modified_care):        
        super(stateEncoder, self).__init__()

        #### parameters ####
        self.hidden_dims_mixtureEnc = encoder_cfg['hidden_dims_mixtureEnc']
        self.num_tasks = encoder_cfg['num_tasks']
        self.use_modified_care = use_modified_care
        if use_modified_care:
            self.input_dim = encoder_cfg['RoBERTa_embedding_dim']
        else:
            self.input_dim = int(encoder_cfg['embedding_dim_contextEnc'])
        #### parameters ####

        #### mixuture of k encoders ####
        self.mixture_encoders = feedForwardEncoder(
            num_encoders=encoder_cfg['num_encoders'],
            input_dim=int(encoder_cfg['state_dim']),
            output_dim=int(encoder_cfg['output_dim_mixtureEnc']),
            hidden_dims=encoder_cfg['hidden_dims_mixtureEnc'] 
        )
        #### mixuture of k encoders ####

        #### trunk is for calculating attention value ####
        self.trunk = build_mlp(
            input_dim=self.input_dim,
            output_dim=int(encoder_cfg['num_encoders']),
            hidden_dims=encoder_cfg['hidden_dims_mixtureEnc'] # same dim as mixture_encoders
        )     
        self.trunk.apply(weights_init)
        #### trunk is for calculating attention value ####

        self.softmax = nn.Softmax(dim=-1)

        if use_modified_care:
            self.mlp_context = build_mlp(
                input_dim=self.input_dim,
                output_dim=int(encoder_cfg['output_dim_contextEnc']),
                hidden_dims=encoder_cfg['hidden_dims_contextEnc'] # same dim as mixture_encoders
            )     
            self.mlp_context.apply(weights_init)

    def mtobss2states_taskIndices(self, mtobss : torch.Tensor) -> (torch.Tensor, torch.Tensor):
        '''
        input :
            mtobss = (batch_size, state_dim+num_tasks)
        output :
            states = (batch_size, state_dim)
            taskIndices : (batch_size, )
        '''
        one_hots = mtobss[:, -self.num_tasks:] # one_hots = (batch_size, num_tasks)
        assert one_hots.shape[1] == self.num_tasks, \
            'The number of tasks does not match self.num_tasks'

        states = mtobss[:, :-self.num_tasks]
        taskIndices = torch.argmax(one_hots, dim=1)

        return states, taskIndices

    def encode_states(self, z_encs, z_context):
        '''
        input :
            z_encs = (batch_size, num_encoders, z_enc_dim)
            z_context = (batch_size, z_context_dim)
                here, z_enc_dim == z_context_dim
        output :
            encoded_states = (batch_size, z_enc_dim+z_context_dim) 
                which will be input of policy algorithm
        '''        
        alpha = self.trunk(z_context.detach()) # alpha = (batch_size, num_encoders)
        alpha = self.softmax(alpha).unsqueeze(dim=-1) # alpha = (batch_size, num_encoders, 1)

        z_enc = (z_encs * alpha).sum(dim=1) # z_enc = (batch_size, z_enc_dim)
        z_enc = z_enc / alpha.sum(dim=1) # z_enc = (batch_size, z_enc_dim)

        if self.use_modified_care:
            z_context = self.mlp_context(z_context)

        encoded_states = torch.cat([z_context, z_enc], dim=1)

        return encoded_states       
    
    def forward(self, z_context : torch.Tensor, mtobss : torch.Tensor, detach_z_encs=False):
        '''
        input :
            z_context = (batch_size, z_context_dim)
            mtobss = (batch_size, mtobs_dim) = (batch_size, state_dim+num_tasks)
                : here, z_enc_dim == z_context_dim if not use modified version of CARE
            detach_z_encs (bool) 
                : Detach output of mixtures of encoders when update actor because
                actor does not update its state encoder, instead, it gets updated
                parameters from hard copy of critic's one. 
                Default is False
        output :
            encoded_states = (batch_size, z_enc_dim+z_context_dim) 
                which will be input of policy algorithm
        '''     
        states, _ = self.mtobss2states_taskIndices(mtobss)

        z_context_dim = z_context.shape[-1]

        z_encs = self.mixture_encoders.forward(states) # (batch_size, num_encoders, z_enc_dim)
        z_enc_dim = z_encs.shape[-1]

        if not self.use_modified_care:
            assert z_context_dim == z_enc_dim, \
                'z_context_dim should be same as z_enc_dim for calculating attention.'

        if detach_z_encs:
            z_encs = z_encs.detach()

        encoded_states = self.encode_states(z_encs, z_context)

        return encoded_states


class Linear(nn.Module):
    def __init__(self,
        num_encoders,
        in_features,
        out_features
    ):
        super(Linear, self).__init__()
        '''
        This is linear layer for parallel computation of `Mixture of Encoders`
        '''
        self.num_encoders = num_encoders
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(
            torch.randn(num_encoders, in_features, out_features), 
            requires_grad=True
        )
        self.b = nn.Parameter(
            torch.randn(num_encoders, 1, out_features), 
            requires_grad=True
        )

    def forward(self, x):
        '''
        input :
            x = (batch_size, in_features) = (b, i)
             or (num_encoders, batch_size, in_features) = (k, b, i)
        output :
            output = (num_encoders, batch_size, out_features) = (k, b, o) for both case of x
        '''
        if len(x.shape) == 2:
            _, in_features = x.shape
            assert in_features == self.in_features, \
                'Error has occured at linear layer for parallel computation of `Mixture of Encoders`'            
            return torch.einsum('kio,bi->kbo', self.W, x) + self.b 
        elif len(x.shape) == 3:
            num_encoders, _, in_features = x.shape
            assert num_encoders == self.num_encoders, \
                'Error has occured at linear layer for parallel computation of `Mixture of Encoders`'
            assert in_features == self.in_features, \
                'Error has occured at linear layer for parallel computation of `Mixture of Encoders`'
            return torch.einsum('kio,kbi->kbo', self.W, x) + self.b 
        else:
            return NotImplementedError()


class feedForwardEncoder(nn.Module):
    def __init__(self,
        num_encoders,
        input_dim,
        output_dim,
        hidden_dims=[50, 50],
    ):
        super(feedForwardEncoder, self).__init__()
        '''
        This is implementation of `Mixture of Encoders`
        '''

        self.mixtureEncoders = nn.ModuleList()

        dims = [input_dim] + hidden_dims
        for in_dim_, out_dim_ in zip(dims[:-1], dims[1:]):
            self.mixtureEncoders.append(
                Linear(
                    num_encoders=num_encoders,
                    in_features=in_dim_,
                    out_features=out_dim_
                )
            )
            self.mixtureEncoders.append(nn.ReLU())

        self.mixtureEncoders.append(
            Linear(
                num_encoders=num_encoders,
                in_features=hidden_dims[-1],
                out_features=output_dim
            )
        )

        self.mixtureEncoders = nn.Sequential(*self.mixtureEncoders)  

    def forward(self, x):
        '''
        input :
            x = (batch_size, input_dim)
        output :
            output = (batch_size, num_encoders, output_dim)
        '''
        return self.mixtureEncoders(x).transpose(1, 0)
