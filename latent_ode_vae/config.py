# neural nets hyperparameters
latent_dim = 10
augmented_dim = 10
encoder_type = 'gru'
encoder_hidden_dim=128
encoder_num_layers=2
encoder_activation='relu'
ode_hidden_dim=64
ode_num_layers=4
ode_activation='tanh'
ode_method='dopri5'
ode_use_adjoint=False
decoder_hidden_dim=32
decoder_num_layers=2
decoder_activation='relu'

# training hyperparameters
num_epochs = 100
num_batches = 100
num_samples_per_batch = 32
learning_rate = 1e-3
sub_length = 50
length_scheduler = True
max_sub_length = 100
min_sub_length = 30
length_step = 20
epoch_step = 20
kl_coeff = 1.0

# dataset
path = f"trajectories/lorenz_gaussian_long_0.05.npz"