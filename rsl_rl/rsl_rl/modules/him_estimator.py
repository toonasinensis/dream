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
                 encoder_hidden_dims=[256, 128],
                #  his_hidden_dims =[512,256],
                 decoder_hidden_dims = [128, 256],
                 latent_dim = 16,
                 activation='elu',
                 learning_rate=1e-3,
                 max_grad_norm=10.0,
                 **kwargs):
        if kwargs:
            print("Estimator_CL.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
        super(HIMEstimator, self).__init__()
        activation = get_activation(activation)

        self.latent_dim = latent_dim
        self.temporal_steps = temporal_steps
        self.num_one_step_obs = num_one_step_obs
        self.max_grad_norm = max_grad_norm
        
        # Encoder

        # Build Decoder  
        decoder_layers = []
        decoder_input_dim = latent_dim + 3
        decoder_layers.append(nn.Sequential(nn.Linear(decoder_input_dim, decoder_hidden_dims[0]),
                                            nn.BatchNorm1d(decoder_hidden_dims[0]),
                                            nn.ELU()))
        for l in range(len(decoder_hidden_dims)):
            if l == len(decoder_hidden_dims) - 1:
                decoder_layers.append(nn.Linear(decoder_hidden_dims[l], num_one_step_obs))
            else:
                decoder_layers.append(nn.Sequential(nn.Linear(decoder_hidden_dims[l], decoder_hidden_dims[l+1]),
                                        nn.BatchNorm1d(decoder_hidden_dims[l+1]),
                                        nn.ELU()))
        self.decoder = nn.Sequential(*decoder_layers)        
        print("decoder",self.decoder)

        #HISTORY Encoder 
        encoder_layers = []
        encoder_input_dim = self.temporal_steps * self.num_one_step_obs
        encoder_layers.append(nn.Sequential(nn.Linear(encoder_input_dim, encoder_hidden_dims[0]),
                                            nn.BatchNorm1d(encoder_hidden_dims[0]),
                                            nn.ELU()))

        encoder_input_dim = self.temporal_steps * self.num_one_step_obs
        for l in range(len(encoder_hidden_dims) - 1):
            encoder_layers.append(nn.Sequential(nn.Linear(encoder_hidden_dims[l], encoder_hidden_dims[l+1]),
                            nn.BatchNorm1d(encoder_hidden_dims[l+1]),
                            nn.ELU()))
        self.encoder = nn.Sequential(*encoder_layers)

        self.fc_mu = nn.Linear(encoder_hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(encoder_hidden_dims[-1], latent_dim)
        self.fc_vel = nn.Linear(encoder_hidden_dims[-1], 3)

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
        eps = torch.clamp(eps, min=-3, max=3)
        return eps * std + mu

    def get_latent(self, obs_history):
        vel, z, _, _ = self.encode(obs_history)
        return vel.detach(), z.detach()

    def forward(self, obs_history):        
        encoded = self.encoder(obs_history.detach())
        vel = self.fc_vel(encoded)
        z = self.fc_mu(encoded)
        return vel.detach(), z.detach()
    
    def decode(self, z, v):
        input = torch.cat([z, v], dim = 1)
        output = self.decoder(input)
        return output

    def encode(self, obs_history):
        encoded = self.encoder(obs_history.detach())
        vel = self.fc_vel(encoded)
        latent_mu = self.fc_mu(encoded)
        latent_logvar = self.fc_var(encoded)
        z = self.reparameterize(latent_mu,latent_logvar) 
        return vel, z, latent_mu, latent_logvar
    
    def update(self, obs_history, critic_obs, next_critic_obs, lr=None, kld_weight = 1.0):
        if lr is not None:
            self.learning_rate = lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate
                
        vel = next_critic_obs[:, self.num_one_step_obs:self.num_one_step_obs+3].detach()
        next_obs = next_critic_obs.detach()[:, 3:self.num_one_step_obs+3]

        vel_pred, z, latent_mu, latent_logvar = self.encode(obs_history)
        # z_t = self.target(next_obs)
        # pred_vel, z_s = z_s[..., :3], z_s[..., 3:]
        estimation_loss = F.mse_loss(vel_pred, vel)
        obs_next_pred =self.decode(z, vel_pred) 
        reconstruct_loss = F.mse_loss(obs_next_pred, next_obs)

        kld_loss = -0.5 * torch.sum(1 + latent_logvar - latent_mu ** 2 - latent_logvar.exp(), dim = 1)
        kld_loss_mean = kld_loss.mean()
        losses = 5.0 * estimation_loss + reconstruct_loss + kld_weight * kld_loss_mean

        self.optimizer.zero_grad()
        losses.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return estimation_loss.item(), reconstruct_loss.item(), kld_loss_mean.item()

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