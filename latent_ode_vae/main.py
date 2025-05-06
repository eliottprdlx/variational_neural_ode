import torch
from dynamics_model import LatentODEVAE
from dataset import Dataset
from utils import train, pretrain_with_length_warmup, plot_imagined_trajectories
from config import *
from masks import FirstN

# detect device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device : {device}")

# initialize dataset
dataset = Dataset(max_size=1000)
dataset.add_from_npz(path)
dataset.normalize()

# infer dimensions
output_dim = dataset.obs_dim
control_dim = dataset.action_dim

# by default full observation is used
input_dim = output_dim

# create mask for partial observation
mask = FirstN(1)
input_dim = mask._dim

print(f"Obs dim: {output_dim} Control dim : {control_dim} Partial obs dim : {input_dim}")
print(f"Encoder type : {encoder_type}")

# special case when encoder_type = identity
if encoder_type == 'id':
    latent_dim = input_dim * sub_length

# create model
model = LatentODEVAE(input_dim, latent_dim, control_dim, augmented_dim, output_dim, device, sub_length, 
                     encoder_type, encoder_hidden_dim, encoder_num_layers, encoder_activation, encoder_learning_rate,
                     ode_hidden_dim, ode_num_layers, ode_activation, ode_method, ode_use_adjoint, ode_learning_rate,
                     decoder_hidden_dim, decoder_num_layers, decoder_activation, 
                     decoder_learning_rate, kl_coeff)
model.to(device)

# train model
if length_warmup and encoder_type in ['gru', 'odegru']:
    print("Pre-training with length warmup ...")
    total, kl, recon = pretrain_with_length_warmup(model, dataset, min_sub_length, max_sub_length, 
                                                   num_batches, num_samples_per_batch, length_step, 
                                                   epoch_step, masker=mask)
    print("Training with full length ...")
    total, kl, recon = train(model, dataset, max_sub_length, num_batches, 
                            num_samples_per_batch, num_epochs, encoder_type, mask)
else:
    total, kl, recon = train(model, dataset, sub_length, num_batches, 
          num_samples_per_batch, num_epochs, encoder_type,
          masker=mask)

plot_imagined_trajectories(model, dataset, sub_length, num_samples=100)


