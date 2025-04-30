import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
from torch.nn.modules.rnn import GRU, LSTM, GRUCell, LSTMCell
from torchdiffeq import odeint as odeint
from diffeq import ODEFunc
import utils


class IdentityEncoder(nn.Module):
    """ z ~ N(μ=x, σ²=1) """
    def __init__(self, input_dim, latent_dim, device):
        super().__init__()
        if input_dim != latent_dim:
            raise ValueError(
                "IdentityEncoder requires input_dim == latent_dim ")
        self.device = device

    def forward(self, x):
        x = x.view(x.size(0), -1)
        mu = x
        logvar = torch.zeros_like(mu)
        return mu, logvar


class LaplaceEncoder(nn.Module):
    """ discrete Laplace / rFFT encoder for inputs of shape (B, T, D) """
    def __init__(self, hidden_dim, latent_dim, num_layers, device, n_freq = None, use_mag_phase = False):
        super().__init__()
        if n_freq is not None:
            n_freq = int(n_freq)            # make sure it’s a plain int
        self.n_freq        = n_freq
        self.use_mag_phase = use_mag_phase
        self.hidden_dim    = hidden_dim
        self.latent_dim    = latent_dim
        self.device = device

        # the three MLP pieces will be built lazily once we know feature size
        self.fc1 = self.fc_mu = self.fc_logvar = None
        self.num_layers = num_layers

    def _build_heads(self, in_dim, device):
        self.fc1 = utils.create_mlp(in_dim, self.hidden_dim,
                                    self.hidden_dim, self.num_layers,
                                    activation='relu').to(device)
        self.fc_mu     = nn.Linear(self.hidden_dim, self.latent_dim).to(device)
        self.fc_logvar = nn.Linear(self.hidden_dim, self.latent_dim).to(device)

    def forward(self, x):
        """
        x : (B, T, D)   — time-major input batch
        returns  μ, log σ²  each (B, latent_dim)
        """
        B, T, D = x.shape                           # unpack shapes

        # bring features to channel axis → (B, D, T)
        x_cd = x.transpose(1, 2)

        # rFFT along time axis (last axis now) → (B, D, F)
        X_fft = torch.fft.rfft(x_cd, dim=-1)        # complex64/128

        # optional low-pass
        if self.n_freq is not None:
            X_fft = X_fft[..., :self.n_freq]

        # build a real-valued feature vector
        if self.use_mag_phase:
            feat = torch.stack([X_fft.abs(), X_fft.angle()], dim=-1).reshape(B, -1)                 # (B, D*F*2)
        else:
            feat = torch.cat([X_fft.real, X_fft.imag], dim=-1).reshape(B, -1)

        # lazily create the MLP heads on the same device as feat
        if self.fc1 is None:
            self._build_heads(feat.size(1), feat.device)

        h      = F.relu(self.fc1(feat))
        mu     = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class MLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers, activation, device):
        super(MLPEncoder, self).__init__()
        self.fc1 = utils.create_mlp(input_dim, hidden_dim, hidden_dim, num_layers, activation='relu')
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


class RNNEncoderBase(nn.Module):
    """ abstract base class for RNN encoders """
    def __init__(self, input_dim, hidden_dim, latent_dim, device):
        super(RNNEncoderBase, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.device = device
    
    def forward(self, x, t=None):
        raise NotImplementedError("Subclasses should implement this method")


class GRUEncoder(RNNEncoderBase):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers, device, bidirectional=True):
        super(GRUEncoder, self).__init__(input_dim, hidden_dim, latent_dim, device)
        self.gru = GRU(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
        multiplier = 2 if bidirectional else 1
        self.fc_mu = nn.Linear(hidden_dim * multiplier, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * multiplier, latent_dim)
        self.bidirectional = bidirectional
    
    def forward(self, x, t=None):
        _, h_n = self.gru(x)  # (num_layers * num_directions, B, H)

        if self.bidirectional:
            forward_final = h_n[-2, :, :]  # shape (B, H)
            backward_final = h_n[-1, :, :] # shape (B, H)
            h_n = torch.cat([forward_final, backward_final], dim=1)  # shape (B, 2*H)
        else:
            h_n = h_n[-1, :, :]  # shape (B, H)

        mu = self.fc_mu(h_n)      # shape (B, latent_dim)
        logvar = self.fc_logvar(h_n)  # shape (B, latent_dim)
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