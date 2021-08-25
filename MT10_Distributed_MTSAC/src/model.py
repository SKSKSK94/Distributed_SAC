import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from utils import weights_init, build_mlp


# continuous action space
class Actor(nn.Module):
    '''
    Params
        actor_cfg : configuration for actor model
    '''
    def __init__(self, actor_cfg, num_tasks):
        super(Actor, self).__init__()
        self.actor_cfg = actor_cfg
        
        self.num_tasks = num_tasks        
        self.state_dim = int(actor_cfg['state_dim'])
        self.mtobs_dim = self.state_dim + num_tasks
        self.action_dim = int(actor_cfg['action_dim'])
        self.action_bound = actor_cfg['action_bound']
        self.k = (self.action_bound[1]-self.action_bound[0])/2

        self.build_model(actor_cfg)
    
    def build_model(self, actor_cfg):
        self.mu_log_std_layer = build_mlp(
            input_dim=self.mtobs_dim,
            output_dim=2*self.action_dim,
            hidden_dims=actor_cfg['actor_hidden_dim']
        )
        self.mu_log_std_layer.apply(weights_init)
        
    def forward(self, mtobss : torch.Tensor):      

        x = self.mu_log_std_layer(mtobss)
        mu, log_std = x[:, :self.action_dim], torch.clamp(x[:, self.action_dim:], -20, 2)
        std = torch.exp(log_std)

        return mu, std    

    def get_action_log_prob_log_std(self, mtobss : torch.Tensor):
        mu, std = self.forward(mtobss=mtobss) # mu = (batch, num_action), std = (batch, num_action)

        ######### log_prob => see the appendix of paper
        normal = Normal(mu, std)
        u = normal.rsample()
        action = self.k*torch.tanh(u)
        gaussian_log_prob = normal.log_prob(u) # (batch, action_dim)
        log_prob = gaussian_log_prob - torch.log(self.k*(1-(action/self.k)**2 + 1e-6)) # (batch, action_dim)
        log_prob = log_prob.sum(dim=-1, keepdim=True) # (batch, 1)

        log_std = torch.log(std)
        
        return action, log_prob, log_std

    def get_action(self, mtobss : torch.Tensor, stochastic=True):
        if not stochastic:
            self.eval()

        mu, std = self.forward(mtobss=mtobss) # mu = (batch, num_action), std = (batch, num_action)

        ######### log_prob => see the appendix of paper
        normal = Normal(mu, std)
        u = normal.rsample()
        action = self.k*torch.tanh(u)

        if not stochastic:
            action = torch.tanh(mu) * self.k
            self.train()

        return action.squeeze(dim=0).detach().cpu().numpy()        

    def cal_loss(self, 
        log_probs, 
        Q_min, 
        alpha,
        use_weighted_loss=False,
        mtobss=None,
        alphas=None
    ):
        '''
        inputs : 
            log_probs = (batch_size, 1)
            Q_min = (batch_size, 1)
            alpha = (batch_size, 1)
            use_weighted_loss (bool)
                calculate loss by weights where low confidence tasks 
                have high weights for training
            mtobss = (batch_size, mtobss_dim)
            alphas = (num_tasks, )
        outputs :
            actor_loss = (tensor)
        '''         
        
        loss = -(Q_min - alpha*log_probs)

        if (
            use_weighted_loss 
            and alphas is not None
            and mtobss is not None
        ):
            assert alphas.shape[0] == self.num_tasks, "alphas shape should be (num_tasks, )"
            one_hots = mtobss[:, -self.num_tasks:] # one_hots = (batch_size, num_tasks)
            task_indicies = torch.argmax(one_hots, dim=1)

            weights = F.softmax(-alphas, dim=0) # (num_tasks, )
            weights = weights[task_indicies].detach()
            weights = weights / weights.sum()

            loss = (weights * loss)

        loss = torch.mean(loss)
        
        return loss


# continuous action space
class Critic(nn.Module):
    '''
    Params
        critic_cfg : configuration for critic model
    '''
    def __init__(self, critic_cfg, num_tasks):
        super(Critic, self).__init__()
        self.critic_cfg = critic_cfg
        
        self.num_tasks = num_tasks
        self.state_dim = int(critic_cfg['state_dim'])
        self.mtobs_dim = self.state_dim + num_tasks
        self.action_dim = int(critic_cfg['action_dim'])

        self.build_model(critic_cfg)

    def build_model(self, critic_cfg):
        self.Q_function_1 = build_mlp(
            input_dim=self.mtobs_dim+self.action_dim,
            output_dim=1,
            hidden_dims=critic_cfg['critic_hidden_dim']
        )
        self.Q_function_1.apply(weights_init)

        self.Q_function_2 = build_mlp(
            input_dim=self.mtobs_dim+self.action_dim,
            output_dim=1,
            hidden_dims=critic_cfg['critic_hidden_dim']
        )
        self.Q_function_2.apply(weights_init)

    def forward(self, mtobss : torch.Tensor, action : torch.Tensor):

        x = torch.cat([mtobss, action], dim=-1)

        return self.Q_function_1(x), self.Q_function_2(x)

    def cal_loss(self, 
        mtobss : torch.Tensor, 
        action,
        td_target_values, 
        use_weighted_loss=False,
        alphas=None,
    ):        
        '''
        inputs : 
            mtobss = (batch_size, mtobss_dim)
            action = (batch_size, action_dim)
            td_target_values = (batch_size, 1)
            use_weighted_loss (bool)
                calculate loss by weights where low confidence tasks 
                have high weights for training
            alphas = (num_tasks, )
        outputs :
            critic_loss = (tensor)
        '''         
        
        current_value_1, current_value_2 = self.forward(mtobss, action)
        loss_1 = (td_target_values - current_value_1)**2
        loss_2 = (td_target_values - current_value_2)**2
        
        if use_weighted_loss and alphas is not None:
            assert alphas.shape[0] == self.num_tasks, "alphas shape should be (num_tasks, )"
            one_hots = mtobss[:, -self.num_tasks:] # one_hots = (batch_size, num_tasks)
            task_indicies = torch.argmax(one_hots, dim=1)

            weights = F.softmax(-alphas, dim=0) # (num_tasks, )
            weights = weights[task_indicies].detach()
            weights = weights / weights.sum()

            loss_1 = (weights * loss_1)
            loss_2 = (weights * loss_2)

        loss_1 = torch.mean(loss_1)
        loss_2 = torch.mean(loss_2)
                        
        return loss_1, loss_2
