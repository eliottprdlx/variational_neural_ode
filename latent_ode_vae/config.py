# neural nets hyperparameters
latent_dim=10
encoder_type = "gru"
encoder_hidden_dim=128
encoder_num_layers=2
encoder_activation='relu'
ode_hidden_dim=64
ode_num_layers=4
ode_activation='tanh'
decoder_hidden_dim=32
decoder_num_layers=2
decoder_activation='relu'

# training hyperparameters
num_epochs = 200
num_batches = 50
num_samples_per_batch = 32
learning_rate = 1e-3
sub_length = 50
max_sub_length = 200
min_sub_length=10
growth_coef=2
length_scheduler = False
kl_coeff = 0.5

# dataset
path = f"trajectories/lorenz_rdsinus.npz"