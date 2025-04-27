import torch

env_name = "CartPole-v1"
seq_len = 100
input_dim = 4  # cartPole state space dimension
hidden_dim = 32
latent_dim = 10
num_layers = 2
num_epochs = 200
batch_size = 32
learning_rate = 0.001
traj_length = 20
device = "cuda" if torch.cuda.is_available() else "cpu"
policy = None
path = f"AC/trajectories/{env_name}_trajectories_lambda.npz"