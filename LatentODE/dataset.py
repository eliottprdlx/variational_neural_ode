from typing import Protocol, List, Tuple
import torch
import numpy as np

class Dataset:
    def __init__(self, max_size: int):
        self.buffer = []
        self.max_size = max_size

    def add_episode(self, traj: List[Tuple]):
        """ traj is a tuple of torch.Tensors (obs, times, actions)"""
        self.buffer.append(traj)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)
    
    def add_from_npz(self, path):
        """ input file must be a .npz with "observations", "times", and "actions" named columns."""
        data = np.load(path, allow_pickle=True)

        for obs, times, actions in zip(data["observations"], data["times"], data["actions"]):
            obs     = torch.as_tensor(np.asarray(obs,     dtype=np.float32))
            times   = torch.as_tensor(np.asarray(times,   dtype=np.float32))
            actions = torch.as_tensor(np.asarray(actions, dtype=np.float32))

            if times.ndim != obs.ndim:
                times = times.unsqueeze(-1)
            if actions.ndim != obs.ndim:
                actions = actions.unsqueeze(-1)

            traj = (obs, times, actions)

            self.buffer.append(traj)

    def sample_subsequences(self, length: int, batch_size: int):
        obs_sub, times_sub, actions_sub = [], [], []
        for _ in range(batch_size):
            n = torch.randint(0, len(self.buffer), (1,)).item()
            obs, times, actions = self.buffer[n]
            T, D = obs.shape
            if T <= length:
                start = 0
            else:
                start = torch.randint(0, T - length + 1, (1,)).item()
            obs_sub.append(obs[start:start + length, :])
            times_sub.append(times[start:start + length, :])
            actions_sub.append(actions[start:start + length, :])

        return (
            torch.stack(obs_sub, dim=0),
            torch.stack(times_sub, dim=0),
            torch.stack(actions_sub, dim=0)
        )