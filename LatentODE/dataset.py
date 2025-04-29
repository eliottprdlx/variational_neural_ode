from typing import Protocol, List, Tuple
import torch
import numpy as np

class Dataset:
    def __init__(self, max_size: int):
        self.buffer = []
        self.max_size = max_size
        self._mean = None
        self._std = None

    def add_episode(self, traj: List[Tuple]):
        """ traj is a tuple of torch.Tensors (obs, times, actions) """
        self.buffer.append(traj)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)
    
    def add_from_npz(self, path):
        """ input file must be a .npz with "observations", "times", and "actions" named columns """
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
    
    def normalize(self):
        if not self.buffer:
            raise ValueError("Buffer is empty – add data before calling.")
        all_obs = torch.cat([traj[0] for traj in self.buffer], dim=0)
        self._mean = all_obs.mean(dim=0)                 # (D,)
        self._std  = all_obs.std (dim=0).clamp_min(1e-8)

    def sample_subsequences(self, length: int, batch_size: int):
        obs_sub, times_sub, actions_sub = [], [], []
        for _ in range(batch_size):
            idx = torch.randint(0, len(self.buffer), (1,)).item()
            obs, times, actions = self.buffer[idx]
            T, _ = obs.shape
            start = 0 if T <= length else torch.randint(0, T - length + 1, ()).item()

            o = obs[start:start + length]
            if self._mean and self._std:
                o = (o - self._mean) / self._std

            obs_sub.append(o)
            times_sub.append(times[start:start + length])
            actions_sub.append(actions[start:start + length])

        return (
            torch.stack(obs_sub, dim=0),       # (B, L, D)
            torch.stack(times_sub, dim=0),     # (B, L, 1)
            torch.stack(actions_sub, dim=0),   # (B, L, A)
        )