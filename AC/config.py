import torch
import gymnasium as gym

def get_env_dt(env):
    """Get the environment's time step (dt)."""
    if hasattr(env.unwrapped, 'tau'):
        return env.unwrapped.tau
    elif hasattr(env.unwrapped, '_dt'):
        return env.unwrapped._dt
    else:
        return 0.05

env_name = "Pendulum-v1"
env = gym.make(env_name)
dt = get_env_dt(env) # time step of the environment 
device="cuda" if torch.cuda.is_available() else "cpu"
episodes= 1000
gamma=0.99 
actor_lr=7e-4 
critic_lr=1e-3 
lambda_=0.9
verbose=False
seq_length= 200
use_gp = True