from generator import Generator
import torch
import gymnasium as gym
from config import *

generator = Generator(env_name, device, episodes, gamma, actor_lr, critic_lr, lambda_, verbose, seq_length, dt, use_gp)
generator.run()
