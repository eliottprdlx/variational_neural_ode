import torch
import gymnasium as gym

# env infos
env_name = "Pendulum-v1"
env = gym.make(env_name)
input_dim = env.observation_space.shape[0]
control_dim = env.action_space.shape[0]

# neural nets spec
hidden_dim = 256
latent_dim = 10
num_layers = 2

# training
num_epochs = 200
num_batches = 30
num_samples_per_batch = 32
learning_rate = 1e-3
traj_length = 50
encoder_type = "gru"


device = "cuda" if torch.cuda.is_available() else "cpu"
policy = None
path = f"AC/trajectories/GP_Pendulum-v1_trajectories.npz"

# TODO : add parameters for every neural nets