import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.rnn import GRU, LSTM, GRUCell, LSTMCell
from torchdiffeq import odeint as odeint
from diffeq import ODEFunc
import utils


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