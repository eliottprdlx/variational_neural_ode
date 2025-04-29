import torch
import numpy as np
from dynamics_model import LatentODEVAE
from dataset import Dataset
from utils import *
from config import *

print(f"Using device : {device}")

print("Creating dataset...")
dataset = Dataset(max_size=1000)
dataset.add_from_npz(path)
# dataset.normalize()

if encoder_type == 'id':
    latent_dim = input_dim * traj_length

print("Dataset created. Initializing model...")
model = LatentODEVAE(input_dim, hidden_dim, latent_dim, control_dim, device, traj_length, encoder_type)
model.to(device)

print("Model created. Training...")
train(model, dataset, traj_length, num_batches, num_samples_per_batch, num_epochs, encoder_type)

print("Training complete. Losses plots saved.")


