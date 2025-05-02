import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
import utils

class MLPDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers, activation, device):
        super(MLPDecoder, self).__init__()
        self.fc1 = utils.create_mlp(latent_dim, hidden_dim, hidden_dim, num_layers, activation)
        self.fc_out = nn.Linear(hidden_dim, input_dim)
        self.device = device
    
    def forward(self, z):
        h = torch.relu(self.fc1(z))
        x_recon = self.fc_out(h)
        return x_recon