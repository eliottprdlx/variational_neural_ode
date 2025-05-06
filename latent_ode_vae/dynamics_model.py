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
    
    @abstractmethod
    def sample(self, N, t, u=None, **kwargs):
        ...


class LatentODEVAE(DynamicsLearner):
    def __init__(
        self,
        input_dim,
        latent_dim,
        control_dim,
        augmented_dim,
        output_dim,
        device,
        traj_length,
        encoder_type='mlp',
        encoder_hidden_dim=64,
        encoder_num_layers=3,
        encoder_activation='relu',
        encoder_learning_rate=1e-3,
        ode_hidden_dim=64,
        ode_num_layers=3,
        ode_activation='tanh',
        ode_method='dopri5',
        ode_use_adjoint=True,
        ode_learning_rate=1e-3,
        decoder_hidden_dim=64,
        decoder_num_layers=3,
        decoder_activation='relu',
        decoder_learning_rate=1e-3,
        kl_coeff=1.0
    ):
        super(LatentODEVAE, self).__init__()
        self.device = device
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.augmented_dim = augmented_dim
        self.control_dim = control_dim
        self.traj_length = traj_length
        self.encoder_learning_rate = encoder_learning_rate
        self.ode_learning_rate = ode_learning_rate
        self.decoder_learning_rate = decoder_learning_rate
        self.kl_coeff = kl_coeff

        # encoder
        if encoder_type == 'mlp':
            self.encoder = MLPEncoder(input_dim * traj_length, encoder_hidden_dim, latent_dim, encoder_num_layers, encoder_activation, device)
        elif encoder_type == 'id':
            self.encoder = IdentityEncoder(input_dim * traj_length, latent_dim, device)
        elif encoder_type == 'laplace':
            self.encoder = LaplaceEncoder(encoder_hidden_dim, latent_dim, encoder_num_layers, device)
        elif encoder_type == 'gru':
            self.encoder = GRUEncoder(input_dim, encoder_hidden_dim, latent_dim, encoder_num_layers, device)
        elif encoder_type == 'odegru':
            self.encoder = ODEGRUEncoder(input_dim, encoder_hidden_dim, latent_dim, encoder_num_layers, encoder_activation, device, method='dopri5')
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")

        # ODE function network
        self.ode_func_net = utils.create_mlp(
            latent_dim + control_dim + augmented_dim,
            ode_hidden_dim,
            latent_dim + augmented_dim,
            ode_num_layers,
            ode_activation
        )
        self.ode_func = ControlledODEFunc(
            self.ode_func_net,
            interp='gaussian',
            interp_kwargs={"sigma": 0.03, "window": 3}
        )

        # ODE solver
        if augmented_dim > 0:
            self.ode_solver = AugmentedDiffEqSolver(self.ode_func, augmented_dim, method=ode_method, use_adjoint=ode_use_adjoint)
        else:
            self.ode_solver = DiffEqSolver(self.ode_func, method=ode_method, use_adjoint=ode_use_adjoint)

        # decoder
        self.decoder = MLPDecoder(input_dim, decoder_hidden_dim, latent_dim, output_dim, decoder_num_layers, decoder_activation, device)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z0 = mu + eps * std
        return z0

    def forward(self, x, t, u, method=None, rtol=None, atol=None):
        mu, logvar = self.encoder(x, t)
        z0 = self.reparameterize(mu, logvar)
        z = self.ode_solver(z0, t, u, method=method, rtol=rtol, atol=atol)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar, z

    def loss_function(self, x_recon, x, mu, logvar, epoch=None):
        recon = F.mse_loss(x_recon, x, reduction='sum') / x.size(0)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        total = recon + self.kl_coeff * kl
        return total, recon, kl
    
    def sample(self, N, t, u=None, method=None, rtol=None, atol=None):
        z0 = torch.randn(N, self.latent_dim).to(self.device)
        z = self.ode_solver(z0, t, u, method=method, rtol=rtol, atol=atol)
        x_recon = self.decoder(z)
        return x_recon, z


class ANODE(DynamicsLearner): ... # TODO : code
