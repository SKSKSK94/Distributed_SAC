import torch
import torch.nn as nn
from torch.distributions import Normal


# continuous action space
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, device, action_bound, hidden_dim=[256, 256], gamma=0.99):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.device = device
        self.action_bound = action_bound

        self.k = (action_bound[1]-action_bound[0])/2

        self.intermediate_dim_list = [state_dim] + hidden_dim

        self.layer_intermediate = nn.ModuleList(
            [nn.Linear(dim_in, dim_out) for dim_in, dim_out in zip(self.intermediate_dim_list[:-1], self.intermediate_dim_list[1:])]
        )        
       
        self.mu_log_std_layer = nn.Linear(self.intermediate_dim_list[-1], 2*self.action_dim)

        self.leak_relu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

        self.apply(self.weights_init_)
    
    def weights_init_(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=1)
            torch.nn.init.constant_(m.bias, 0)
    
    def forward(self, state):        
        x = state     
        
        for linear_layer in self.layer_intermediate:
            x = self.relu(linear_layer(x))

        x = self.mu_log_std_layer(x)
        mu, log_std = x[:, :self.action_dim], torch.clamp(x[:, self.action_dim:], -20, 2)
        std = torch.exp(log_std)

        return mu, std    

    def get_action_log_prob(self, state, stochstic=True):
        mu, std = self.forward(state) # mu = (batch, num_action), std = (batch, num_action)

        ######### log_prob => see the appendix of paper
        normal = Normal(mu, std)
        u = normal.rsample()
        action = self.k*torch.tanh(u)
        gaussian_log_prob = normal.log_prob(u) # (batch, action_dim)
        log_prob = gaussian_log_prob - torch.log(self.k*(1-(action/self.k)**2 + 1e-6)) # (batch, action_dim)
        log_prob = log_prob.sum(dim=-1, keepdim=True) # (batch, 1)
        
        if not stochstic:
            # action = np.clip(mu.detach().cpu().numpy(), self.action_bound[0], self.action_bound[1])
            action = torch.tanh(mu).detach().cpu().numpy() * self.k

        return action, log_prob  

    def get_action(self, state, stochastic=True):
        if not stochastic:
            self.eval()

        mu, std = self.forward(state) # mu = (batch, num_action), std = (batch, num_action)

        ######### log_prob => see the appendix of paper
        normal = Normal(mu, std)
        u = normal.rsample()
        action = self.k*torch.tanh(u)

        if not stochastic:
            # action = np.clip(mu.detach().cpu().numpy(), self.action_bound[0], self.action_bound[1])
            action = torch.tanh(mu).detach().cpu().numpy() * self.k

        return action        

    def cal_loss(self, log_probs, Q_min, alpha):

        loss = -torch.mean(Q_min - alpha*log_probs)
        
        return loss


# continuous action space
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, device, hidden_dim=[256, 256], gamma=0.99):
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.device = device
        
        self.dim_list = hidden_dim + [1]

        self.first_layer = nn.Linear(in_features=state_dim+action_dim, out_features=hidden_dim[0])

        self.layer_module = nn.ModuleList(
            [nn.Linear(dim_in, dim_out) for dim_in, dim_out in zip(self.dim_list[:-1], self.dim_list[1:])]
        )
        
        self.activation = nn.ReLU()

        self.apply(self.weights_init_)

    def weights_init_(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=1)
            torch.nn.init.constant_(m.bias, 0)

    def forward(self, state, action):
        
        x = torch.cat([state, action], dim=-1)
        x = self.activation(self.first_layer(x))
        
        for layer in self.layer_module[:-1]: # not include out layer
            x = self.activation(layer(x))

        x = self.layer_module[-1](x)

        return x

    def cal_loss(self, states, actions, td_target_values):        
        '''
        inputs : 
            state = (batch_size, state_dim)
            action = (batch_size, action_dim)
            td_target_values = (batch_size, 1)
        outputs :
            critic_loss (float)
        '''         

        current_value = self.forward(states, actions)
        loss = torch.mean((td_target_values - current_value)**2)
        
        return loss
