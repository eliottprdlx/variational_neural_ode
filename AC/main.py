from trainer import Trainer
import torch
import gymnasium as gym
from config import *

trainer = Trainer(env_name, device, episodes, gamma, actor_lr, critic_lr, lambda_, verbose, seq_length, dt)
trainer.run()
