import torch
import torch.nn as nn
import torch.nn.functional as F
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import numpy as np
from env import Dataset


def create_mlp(input_dim, hidden_dim, output_dim, num_layers=2, activation='relu'):
    layers = []
    layers.append(nn.Linear(input_dim, hidden_dim))
    
    for _ in range(num_layers - 1):
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        elif activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        
        layers.append(nn.Linear(hidden_dim, hidden_dim))
    
    layers.append(nn.Linear(hidden_dim, output_dim))
    
    return nn.Sequential(*layers)

def sample_sub_trajectories(obs, times, actions, sub_length, n_samples):
    B, T,  D = obs.shape
    obs_sub, times_sub, actions_sub = [], [], []

    for _ in range(n_samples):
        b = torch.randint(0, B, (1,)).item()
        if T <= sub_length:
            start = 0
        else:
            start = torch.randint(0, T - sub_length + 1, (1,)).item()
        obs_sub.append(obs[b, start:start + sub_length, :])
        times_sub.append(times[b, start:start + sub_length, :])
        actions_sub.append(actions[b, start:start + sub_length, :])

    return (
        torch.stack(obs_sub, dim=0),
        torch.stack(times_sub, dim=0),
        torch.stack(actions_sub, dim=0)
    )

def train(model, obs, times, actions, sub_length=60, n_samples=20, num_epochs=1000):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=model.learning_rate)
    total_losses, recon_losses, kl_losses = [], [], []

    if times.ndim != obs.ndim:
        times = times.unsqueeze(-1)
    if actions.ndim != obs.ndim:
        actions = actions.unsqueeze(-1)

    for epoch in range(num_epochs):
        # sample a global batch of sub-trajectories
        obs_sub, times_sub, actions_sub = sample_sub_trajectories(obs, times, actions, sub_length, n_samples)
        obs_sub   = obs_sub.to(model.device)
        times_sub = times_sub.to(model.device)
        actions_sub = actions_sub.to(model.device)

        # forward pass
        x_recon, mu, logvar, z = model(obs_sub, times_sub, actions_sub)

        # compute loss
        total_loss, recon_loss, kl_loss = model.loss_function(x_recon, obs_sub, mu, logvar)

        # backward
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        total_losses.append(total_loss.item())
        recon_losses.append(recon_loss.item())
        kl_losses.append(kl_loss.item())

        if epoch % 20 == 0:
            print(f"Epoch {epoch}/{num_epochs}, Loss: {total_loss.item():.4f}, Recon Loss: {recon_loss.item():.4f}, KL Loss: {kl_loss.item():.4f}")

    return total_losses, recon_losses, kl_losses

def plot_losses(total_losses, recon_losses, kl_losses):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=total_losses, mode='lines', name='Total Loss'))
    fig.add_trace(go.Scatter(y=recon_losses, mode='lines', name='Reconstruction Loss'))
    fig.add_trace(go.Scatter(y=kl_losses, mode='lines', name='KL Divergence Loss'))
    fig.update_layout(title='Losses during Training', xaxis_title='Epochs', yaxis_title='Loss')
    fig.show()
    fig.write_html("LatentODE/plots/losses.html")

def plot_latent_space(model, data, num_samples=1000):
    model.eval()
    with torch.no_grad():
        mu, logvar = model.encoder(data[:num_samples])
        z = model.reparameterize(mu, logvar)
    z = pca.fit_transform(z.cpu().numpy())
    fig = go.Figure(data=[go.Scatter(x=z[:, 0].cpu(), y=z[:, 1].cpu(), mode='markers')])
    fig.update_layout(title='Latent Space Representation', xaxis_title='Latent Dimension 1', yaxis_title='Latent Dimension 2')
    fig.show()
    fig.write_html("LatentODE/plots/latent_space.html")

def create_obs_times_actions_from_path(path, device):
    dataset = np.load(path, allow_pickle=True)
    obs_array = np.stack(dataset["observations"].tolist(), axis=0).astype(np.float32)
    observations = torch.tensor(obs_array, dtype=torch.float32).to(device)
    times_array = np.stack(dataset["times"].tolist(), axis=0).astype(np.float32)
    times = torch.tensor(times_array, dtype=torch.float32).to(device)
    actions_array = np.stack(dataset["actions"].tolist(), axis=0).astype(np.float32)
    actions = torch.tensor(actions_array, dtype=torch.float32).to(device)
    return observations, times, actions

def create_obs_times_actions_from_env(env_name, policy, device):
    data = Dataset(env_name, seq_len, dt, policy=policy)
    observations, times, actions = data.sample_trajectory()
    observations = torch.tensor(observations, dtype=torch.float32).to(device)
    times = torch.tensor(times, dtype=torch.float32).to(device) 
    actions = torch.tensor(actions, dtype=torch.float32).to(device)