import torch
import numpy as np
from models import LatentODEVAE
from utils import *
from config import *

print("Creating dataset...")
if path:
    observations, times, actions = create_obs_times_actions_from_path(path, device)
else:
    observations, times, actions = create_obs_times_actions_from_env(env_name, policy, device)
control_dim = actions.shape[-1] if actions.ndim == 3 else 1

print("Dataset created. Initializing model...")
model = LatentODEVAE(input_dim, hidden_dim, latent_dim, control_dim, device, traj_length, encoder_type="gru")
model.to(device)

print("Model created. Training...")
total_losses, recon_losses, kl_losses = train(model, observations, times, actions, traj_length, num_batches, num_samples_per_batch, num_epochs)
plot_losses(total_losses, recon_losses, kl_losses)

print("Training complete. Losses and latent space plots saved.")


