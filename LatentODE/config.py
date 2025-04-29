import torch

env_name = "CartPole-v1"
input_dim = 4  # cartPole state space dimension
control_dim = 1
hidden_dim = 128
latent_dim = 10
num_layers = 2
num_epochs = 500
num_batches = 20
num_samples_per_batch = 128
learning_rate = 1e-3
traj_length = 10
encoder_type = "gru"
device = "cuda" if torch.cuda.is_available() else "cpu"
policy = None
path = f"AC/trajectories/{env_name}_trajectories_lambda.npz"

# TODO : add parameters for every neural nets