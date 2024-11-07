import copy
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as torchd
from torch.distributions import Normal, Categorical


class HIMEstimator(nn.Module):
    def __init__(self,
                 temporal_steps,
                 num_one_step_obs,
                 enc_hidden_dims=[128, 64, 16],
                 tar_hidden_dims=[128, 64],
                 his_hidden_dims =[512,256],
                 latent_dims = 16,
                 activation='elu',
                 learning_rate=1e-3,
                 max_grad_norm=10.0,
                 num_prototype=32,
                 temperature=3.0,
                 **kwargs):
        if kwargs:
            print("Estimator_CL.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
        super(HIMEstimator, self).__init__()
        activation = get_activation(activation)

        self.latent_dims = latent_dims
        self.temporal_steps = temporal_steps
        self.num_one_step_obs = num_one_step_obs
        self.num_latent = enc_hidden_dims[-1]
        self.max_grad_norm = max_grad_norm
        self.temperature = temperature

        # Encoder



        # enc_input_dim = self.temporal_steps * self.num_one_step_obs
        # enc_layers = []
        # for l in range(len(enc_hidden_dims) - 1):
        #     enc_layers += [nn.Linear(enc_input_dim, enc_hidden_dims[l]), activation]
        #     enc_input_dim = enc_hidden_dims[l]
        # enc_layers += [nn.Linear(enc_input_dim, enc_hidden_dims[-1] + 3)]
        # self.encoder = nn.Sequential(*enc_layers)


        #HISTORY Encoder 
        enc_input_dim = self.temporal_steps * self.num_one_step_obs
        HIS_enc_layers = []
        for l in range(len(his_hidden_dims) - 1):
            HIS_enc_layers += [nn.Linear(enc_input_dim, enc_hidden_dims[l]), activation]
            enc_input_dim = enc_hidden_dims[l]
        HIS_enc_layers += [nn.Linear(enc_input_dim, latent_dims*4)]#why times 4
        self.encoder = nn.Sequential(*HIS_enc_layers)

        self.vel_mu = nn.Linear(latent_dims * 4, 3)
        self.vel_logvar = nn.Linear(latent_dims * 4, 3)

        self.latent_mu = nn.Linear(latent_dims * 4, latent_dims)
        self.latent_logvar = nn.Linear(latent_dims * 4, latent_dims)
        # Optimizer
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
    


    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def get_latent(self, obs_history):
        vel, z = self.encode(obs_history)
        return vel.detach(), z.detach()

    def forward(self, obs_history):
        vel, z = self.encode(obs_history)
        return vel.detach(), z.detach()

    def encode(self, obs_history):
        encoded = self.encoder(obs_history.detach())
        vel_mu = self.vel_mu(encoded)
        vel_logvar = self.vel_logvar(encoded)
        vel = self.reparameterize(vel_mu, vel_logvar)
        latent_mu = self.latent_mu(encoded)
        latent_logvar = self.latent_logvar(encoded)
        z = self.reparameterize(latent_mu,latent_logvar) 
        return vel, z

    def update(self, obs_history, next_critic_obs, lr=None):
        if lr is not None:
            self.learning_rate = lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate
                
        vel = next_critic_obs[:, self.num_one_step_obs:self.num_one_step_obs+3].detach()
        next_obs = next_critic_obs.detach()[:, 3:self.num_one_step_obs+3]

        vel_pred,obs_next_pred= self.encode(obs_history)
        # z_t = self.target(next_obs)
        # pred_vel, z_s = z_s[..., :3], z_s[..., 3:]
        estimation_loss = F.mse_loss(vel_pred, vel)
        losses = estimation_loss#+ swap_loss

        self.optimizer.zero_grad()
        losses.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return estimation_loss.item(), 0





def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "silu":
        return nn.SiLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None