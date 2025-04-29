import torch

env_name = "CartPole-v1"
input_dim = 4  # cartPole state space dimension
control_dim = 1
hidden_dim = 256
latent_dim = 10
num_layers = 2
num_epochs = 200
num_batches = 10
num_samples_per_batch = 64
learning_rate = 1e-3
traj_length = 50
encoder_type = "gru"
device = "cuda" if torch.cuda.is_available() else "cpu"
policy = None
path = f"AC/trajectories/{env_name}_trajectories_lambda.npz"

# TODO : add parameters for every neural nets