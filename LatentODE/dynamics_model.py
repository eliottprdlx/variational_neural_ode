from abc import abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import kl_divergence
from encoders import *
from decoders import *
from diffeq import *
import utils


# abstract class for dynamics learning
class DynamicsLearner(nn.Module):
    @abstractmethod
    def forward(self, x, t, u=None, **kwargs):
        ...

    @abstractmethod
    def loss_function(self, x_recon, x, *extras):
        ...


class LatentODEVAE(DynamicsLearner):
    def __init__(self, input_dim, hidden_dim, latent_dim, control_dim, device, traj_length, encoder_type='mlp', nonlinear_func=None, lr=1e-3):
        super(LatentODEVAE, self).__init__()
        self.device = device
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.control_dim = control_dim
        self.traj_length = traj_length
        self.learning_rate = lr
        num_layers = 3 # TODO : separate num_layers for each neural nets
        
        if encoder_type == 'mlp':
            self.encoder = MLPEncoder(input_dim*traj_length, hidden_dim, latent_dim, num_layers, device)
        elif encoder_type == 'id':
            self.encoder = IdentityEncoder(input_dim*traj_length, latent_dim, device)
        elif encoder_type == 'laplace':
            self.encoder = LaplaceEncoder(hidden_dim, latent_dim, num_layers, device)
        elif encoder_type == 'gru':
            self.encoder = GRUEncoder(input_dim, hidden_dim, latent_dim, num_layers, device)
        elif encoder_type == 'lstm':
            self.encoder = LSTMEncoder(input_dim, hidden_dim, latent_dim, device)
        elif encoder_type == 'odegru':
            self.encoder = ODEGRUEncoder(input_dim, hidden_dim, latent_dim, device, method='dopri5', rtol=None, atol=None)
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")
        
        self.ode_func_net = utils.create_mlp(latent_dim + control_dim, hidden_dim, latent_dim, num_layers, activation='relu')
        self.ode_func = ControlledODEFunc(self.ode_func_net, nonlinear_func, interp='gaussian', interp_kwargs={"sigma": 0.03, "window": 3} )
        self.ode_solver = DiffEqSolver(self.ode_func, method="dopri5")
        
        self.decoder = MLPDecoder(input_dim, hidden_dim, latent_dim, num_layers, device)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z0 = mu + eps * std
        return z0
    
    def forward(self, x, t, u, method=None, rtol=None, atol=None):
        mu, logvar = self.encoder(x)
        z0 = self.reparameterize(mu, logvar)
        z = self.ode_solver(z0, t, u, method=method, rtol=rtol, atol=atol)
        z = z.permute(1, 0, 2)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar, z
    
    def loss_function(self, x_recon, x, mu, logvar, epoch=None, beta_max=1.0, kl_anneal_epochs=100):
        # Reconstruction loss
        recon = F.mse_loss(x_recon, x, reduction='mean')

        # KL divergence
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

        # Compute annealed beta
        if epoch is None or kl_anneal_epochs == 0:
            beta = beta_max  # no annealing
        else:
            beta = beta_max * min(1.0, epoch / kl_anneal_epochs)

        total = recon + beta * kl
        return total, recon, kl



class ANODE(DynamicsLearner): ... # TODO : code
