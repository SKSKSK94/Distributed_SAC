import torch
import torch.nn as nn
from torch.distributions import Normal
from utils import weights_init, build_mlp

from state_encoder import stateEncoder


# continuous action space
class Actor(nn.Module):
    def __init__(self, 
            actor_cfg,  
            encoder_cfg, 
        ):
        super(Actor, self).__init__()
        self.actor_cfg = actor_cfg
        self.encoder_cfg = encoder_cfg
        
        self.state_dim = int(actor_cfg['state_dim'])
        self.mtobs_dim = self.state_dim + encoder_cfg['num_tasks']
        self.policy_input_dim = encoder_cfg['output_dim_contextEnc'] + encoder_cfg['output_dim_mixtureEnc']
        self.action_dim = int(actor_cfg['action_dim'])
        self.action_bound = actor_cfg['action_bound']
        self.k = (self.action_bound[1]-self.action_bound[0])/2

        self.state_encoder = stateEncoder(encoder_cfg)

        self.mu_log_std_layer = build_mlp(
            input_dim=self.policy_input_dim,
            output_dim=2*self.action_dim,
            hidden_dims=actor_cfg['actor_hidden_dim']
        )
        self.mu_log_std_layer.apply(weights_init)
        
    def forward(self, 
        mtobss : torch.Tensor, 
        z_context : torch.Tensor, 
        detach_z_encs=False
    ):      
        encoded_state = self.state_encoder.forward(
            z_context=z_context,
            mtobss=mtobss,
            detach_z_encs=detach_z_encs
        )

        x = self.mu_log_std_layer(encoded_state)
        mu, log_std = x[:, :self.action_dim], torch.clamp(x[:, self.action_dim:], -20, 2)
        std = torch.exp(log_std)

        return mu, std    

    def get_action_log_prob_log_std(self, 
        mtobss : torch.Tensor, 
        z_context : torch.Tensor, 
        detach_z_encs=False, 
    ):
        mu, std = self.forward(
            z_context=z_context,
            mtobss=mtobss,
            detach_z_encs=detach_z_encs
        ) # mu = (batch, num_action), std = (batch, num_action)

        ######### log_prob => see the appendix of paper
        normal = Normal(mu, std)
        u = normal.rsample()
        action = self.k*torch.tanh(u)
        gaussian_log_prob = normal.log_prob(u) # (batch, action_dim)
        log_prob = gaussian_log_prob - torch.log(self.k*(1-(action/self.k)**2 + 1e-6)) # (batch, action_dim)
        log_prob = log_prob.sum(dim=-1, keepdim=True) # (batch, 1)

        log_std = torch.log(std)
        
        return action, log_prob, log_std

    def get_action(self, 
        mtobss : torch.Tensor, 
        z_context : torch.Tensor, 
        detach_z_encs=False, 
        stochastic=True
    ):
        if not stochastic:
            self.eval()

        with torch.no_grad():
            mu, std = self.forward(
                z_context=z_context,
                mtobss=mtobss,
                detach_z_encs=detach_z_encs
            ) # mu = (batch, num_action), std = (batch, num_action)

            ######### log_prob => see the appendix of paper
            normal = Normal(mu, std)
            u = normal.rsample()
            action = self.k*torch.tanh(u)

        if not stochastic:
            action = torch.tanh(mu) * self.k

        return action.squeeze(dim=0).detach().cpu().numpy()        

    def cal_loss(self, log_probs, Q_min, alpha):

        loss = -torch.mean(Q_min - alpha*log_probs)
        
        return loss


# continuous action space
class Critic(nn.Module):
    def __init__(self, 
            critic_cfg,  
            encoder_cfg, 
        ):
        super(Critic, self).__init__()
        self.critic_cfg = critic_cfg
        self.encoder_cfg = encoder_cfg
        
        self.state_dim = int(critic_cfg['state_dim'])
        self.mtobs_dim = self.state_dim + encoder_cfg['num_tasks']
        self.policy_input_dim = encoder_cfg['output_dim_contextEnc'] + encoder_cfg['output_dim_mixtureEnc']
        self.action_dim = int(critic_cfg['action_dim'])

        self.build_model(encoder_cfg, critic_cfg)

    def build_model(self, encoder_cfg, critic_cfg):
        self.state_encoder = stateEncoder(encoder_cfg)

        self.Q_function_1 = build_mlp(
            input_dim=self.policy_input_dim+self.action_dim,
            output_dim=1,
            hidden_dims=critic_cfg['critic_hidden_dim']
        )
        self.Q_function_1.apply(weights_init)

        self.Q_function_2 = build_mlp(
            input_dim=self.policy_input_dim+self.action_dim,
            output_dim=1,
            hidden_dims=critic_cfg['critic_hidden_dim']
        )
        self.Q_function_2.apply(weights_init)

    def forward(self, 
        mtobss : torch.Tensor, 
        z_context : torch.Tensor, 
        action : torch.Tensor, 
        detach_z_encs=False
    ):
        encoded_state = self.state_encoder.forward(
            z_context=z_context,
            mtobss=mtobss,
            detach_z_encs=detach_z_encs
        )

        x = torch.cat([encoded_state, action], dim=-1)

        return self.Q_function_1(x), self.Q_function_2(x)

    def cal_loss(self, 
        mtobss : torch.Tensor, 
        z_context : torch.Tensor, 
        action,
        td_target_values, 
        detach_z_encs=False
    ):        
        '''
        inputs : 
            mtobss = (batch_size, mtobss_dim)
            z_context = (batch_size, z_context_dim)
            action = (batch_size, action_dim)
            td_target_values = (batch_size, 1)
            detach_z_encs (bool)
        outputs :
            critic_loss = (tensor)
        '''         

        current_value_1, current_value_2 = self.forward(mtobss, z_context, action, detach_z_encs)
        loss_1 = torch.mean((td_target_values - current_value_1)**2)
        loss_2 = torch.mean((td_target_values - current_value_2)**2)
        
        return loss_1, loss_2
