import torch
from dynamics_model import LatentODEVAE
from dataset import Dataset
from utils import train, train_with_length_scheduler
from config import *

# detect device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device : {device}")

# initialize dataset
dataset = Dataset(max_size=1000)
dataset.add_from_npz(path)
dataset.normalize()

# infer dimensions
input_dim = dataset.obs_dim
control_dim = dataset.action_dim
print(f"Obs dim: {input_dim} Control dim : {control_dim}")

# special case when encoder_type = identity
if encoder_type == 'id':
    latent_dim = input_dim * sub_length

# create model
model = LatentODEVAE(input_dim, latent_dim, control_dim, device, sub_length, encoder_type,
    encoder_hidden_dim, encoder_num_layers, encoder_activation, ode_hidden_dim, ode_num_layers, 
    ode_activation, decoder_hidden_dim, decoder_num_layers, decoder_activation, kl_coeff)
model.to(device)

# train model
if length_scheduler and encoder_type in ['gru', 'odegru']:
    train_with_length_scheduler(model, dataset, max_sub_length, num_batches, num_samples_per_batch, num_epochs, encoder_type)
else:
    train(model, dataset, sub_length, num_batches, num_samples_per_batch, num_epochs, encoder_type,
                                min_sub_length, growth_coef)


