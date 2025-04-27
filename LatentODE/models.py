import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import kl_divergence
from torch.nn.modules.rnn import GRU, LSTM, GRUCell, LSTMCell
from torchdiffeq import odeint as odeint
import utils


class ODEFunc(nn.Module):
    def __init__(self, ode_func_net, nonlinear_func = None):
        super(ODEFunc, self).__init__()
        self.ode_func_net = ode_func_net
        self.nonlinear_func = nonlinear_func

    def forward(self, t, z):
        out = self.ode_func_net(z)
        return self.nonlinear_func(out) if self.nonlinear_func else out


class ControlledODEFunc(ODEFunc):
    def __init__(self, ode_func_net, nonlinear_func = None):
        super(ControlledODEFunc, self).__init__(ode_func_net, nonlinear_func)
        self.u: torch.Tensor    # shape (batch, T, u_dim)
        self.times: torch.Tensor  # shape (T,)

    def forward(self, t, z):
        idx = torch.argmin(torch.abs(self.times - t)).item()
        u_t = self.u[:, idx, :]  # (batch, u_dim)
        z_and_u = torch.cat([z, u_t], dim=-1)
        in_feats = self.ode_func_net[0].in_features
        out = self.ode_func_net(z_and_u)
        return self.nonlinear_func(out) if self.nonlinear_func else out


class DiffEqSolver(nn.Module):
    def __init__(self, ode_func, method = 'dopri5', rtol = None, atol = None):
        super(DiffEqSolver, self).__init__()
        self.ode_func = ode_func
        self.method = method
        self.rtol = rtol
        self.atol = atol

    def forward(self, z0, t, u = None,method = None,rtol = None,atol = None):
        method = method or self.method
        rtol   = rtol   or self.rtol
        atol   = atol   or self.atol
        t = t.squeeze(-1)
        # t: (B,T) or (T,)
        if t.ndim == 2:
            # assume every row t[i] is the same
            # you could also sanity-check with torch.allclose
            t = t[0]

        if u is not None:
            # u: (B,T,u_dim)
            self.ode_func.u     = u
            self.ode_func.times = t

        # one batched call: z0 is (B,latent), t is (T,)
        # returns (T, B, latent)
        z = odeint(self.ode_func, z0, t, method=method)
        return z


class MLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, device):
        super(MLPEncoder, self).__init__()
        self.fc1 = utils.create_mlp(input_dim, hidden_dim, hidden_dim, num_layers=2, activation='relu')
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.device = device
    
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        h = torch.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Conv1DEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, device):
        super(Conv1DEncoder, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, stride=2)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.device = device
    
    def forward(self, x):
        h = torch.relu(self.conv1(x))
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class RNNEncoderBase(nn.Module):
    """abstract base class for RNN encoders"""
    def __init__(self, input_dim, hidden_dim, latent_dim, device):
        super(RNNEncoderBase, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.device = device
    
    def forward(self, x, t=None):
        raise NotImplementedError("Subclasses should implement this method")


class GRUEncoder(RNNEncoderBase):
    def __init__(self, input_dim, hidden_dim, latent_dim, device):
        super(GRUEncoder, self).__init__(input_dim, hidden_dim, latent_dim, device)
        self.gru = GRU(input_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, x, t=None):
        _, h_n = self.gru(x)       # h_n: (1, batch, hidden_dim)
        h_n = h_n.squeeze(0)       # → (batch, hidden_dim)
        mu     = self.fc_mu(h_n)   # → (batch, latent_dim)
        logvar = self.fc_logvar(h_n)
        return mu, logvar


class LSTMEncoder(RNNEncoderBase):
    def __init__(self, input_dim, hidden_dim, latent_dim, device):
        super(LSTMEncoder, self).__init__(input_dim, hidden_dim, latent_dim, device)
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, x, t=None):
        _, (h_n, _) = self.lstm(x)
        mu = self.fc_mu(h_n[-1])
        logvar = self.fc_logvar(h_n[-1])
        return mu, logvar


class ODEGRUEncoder(RNNEncoderBase):
    """encoder that alternates between GRU updates and ODE evolution.
    based on the paper "Latent Ordinary Differential Equations for Irregularly-Sampled Time Series" by
    Rubanova et al. (2019). particularly useful for irregularly sampled time series data."""
    def __init__(self, input_dim, hidden_dim, latent_dim, device, method='dopri5', rtol=1e-5, atol=1e-5):
        super(ODEGRUEncoder, self).__init__(input_dim, hidden_dim, latent_dim, device)
        self.ode_func = ODEFunc(hidden_dim)
        self.gru_cell = nn.GRUCell(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.ode_method = method
        self.ode_rtol = rtol
        self.ode_atol = atol

    def forward(self, x, t):
        batch_size, seq_len, _ = x.size()

        # initialize hidden state h_0 to zeros
        h = torch.zeros(batch_size, self.hidden_dim, device=self.device)

        # initial GRU update at first observation time
        h = self.gru_cell(x[:, 0, :], h)

        # loop through sequence, ODE evolve then GRU update
        for i in range(1, seq_len):
            t0, t1 = t[i - 1], t[i]
            t_tensor = torch.tensor([t0, t1], dtype=torch.float32, device=self.device)
            # evolve hidden state via ODE
            h = odeint(self.ode_func, h, t_tensor, atol=self.ode_atol, rtol=self.ode_rtol, method=self.ode_method)[-1]
            # GRU update with new observation
            h = self.gru_cell(x[:, i, :], h)

        # project final hidden state to latent distribution parameters
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, device):
        super(Decoder, self).__init__()
        self.fc1 = utils.create_mlp(latent_dim, hidden_dim, hidden_dim, num_layers=2, activation='relu')
        self.fc_out = nn.Linear(hidden_dim, input_dim)
        self.device = device
    
    def forward(self, z):
        h = torch.relu(self.fc1(z))
        x_recon = self.fc_out(h)
        return x_recon


class LatentODEVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, control_dim, device, traj_length, encoder_type='mlp', nonlinear_func=None, lr=1e-3):
        super(LatentODEVAE, self).__init__()
        self.device = device
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.control_dim = control_dim
        self.traj_length = traj_length
        self.learning_rate = lr
        
        if encoder_type == 'mlp':
            self.encoder = MLPEncoder(input_dim*traj_length, hidden_dim, latent_dim, device)
        elif encoder_type == 'conv1d':
            self.encoder = Conv1DEncoder(input_dim, hidden_dim, latent_dim, device)
        elif encoder_type == 'gru':
            self.encoder = GRUEncoder(input_dim, hidden_dim, latent_dim, device)
        elif encoder_type == 'lstm':
            self.encoder = LSTMEncoder(input_dim, hidden_dim, latent_dim, device)
        elif encoder_type == 'odegru':
            self.encoder = ODEGRUEncoder(input_dim, hidden_dim, latent_dim, device, method='dopri5', rtol=None, atol=None)
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")
        self.ode_func_net = utils.create_mlp(latent_dim + control_dim, hidden_dim, latent_dim, num_layers=2, activation='relu')
        self.ode_func = ControlledODEFunc(self.ode_func_net, nonlinear_func=nonlinear_func)
        self.ode_solver = DiffEqSolver(self.ode_func)
        
        self.decoder = Decoder(input_dim, hidden_dim, latent_dim, device)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z0 = mu + eps * std
        return z0
    
    def forward(self, x, t, u=None, method=None, rtol=None, atol=None):
        mu, logvar = self.encoder(x)
        z0 = self.reparameterize(mu, logvar)
        z = self.ode_solver(z0, t, u, method=method, rtol=rtol, atol=atol)
        x_recon = self.decoder(z)
        x_recon = x_recon.permute(1, 0, 2)
        return x_recon, mu, logvar, z
    
    def loss_function(self, x_recon, x, mu, logvar):
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_loss, recon_loss, kl_loss
