"""
generator.py
============

TrajectoryGenerator now supports two modes:

1. mode="multi"
     Simulate n_traj separate trajectories, each of length traj_length,
     starting from n_traj different initial conditions.

2. mode="long"
     Simulate ONE long trajectory of length n_traj * traj_length starting
     from x0, then slice it into n_traj consecutive segments of length
     traj_length.

The extract_obs_time_action function is unchanged.
"""
import random
from typing import List, Tuple, Literal, Sequence, Optional

import numpy as np
import torch  # only for tensor -> numpy conversion

from dynamics import Dynamics
from controls import ControlPolicy

from tqdm import tqdm


class TrajectoryGenerator:
    def __init__(
        self,
        dynamics: Dynamics,
        policy: ControlPolicy,
        dt: float = 0.02,
        seed: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        dynamics   : instance of a Dynamics subclass
        policy     : instance of a ControlPolicy subclass
        dt         : integration step in seconds
        seed       : random seed (None means do not touch rng state)
        """
        self.dyn = dynamics
        self.policy = policy
        self.dt = dt

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    # ------------------------------------------------------------------
    # Main trajectory generator
    # ------------------------------------------------------------------
    def rollout(
        self,
        x0: np.ndarray,
        n_traj: int,
        traj_length: float,
        mode: Literal["multi", "long"] = "multi",
        start_points: Optional[Sequence[np.ndarray]] = None,
        init_noise_std: float = 0.0,
        action_noise_std: float = 0.0,
    ) -> List[List[Tuple[np.ndarray, np.ndarray, float]]]:
        """
        Generate trajectories under the requested mode.

        Parameters
        ----------
        x0              : reference initial state
        n_traj          : number of trajectories wanted
        traj_length     : length of EACH trajectory in seconds
        mode            : "multi" or "long" (see top of file)
        start_points    : optional list/array of initial states for
                          mode="multi" (length must equal n_traj)
        init_noise_std  : std dev of Gaussian noise added to each start state
                          (ignored if start_points is supplied)
        action_noise_std: std dev of Gaussian noise added to every control
                          output at each step

        Returns
        -------
        trajectories : list where each element is a list of
                       (state, action, time) tuples
        """
        steps_per_traj = int(round(traj_length / self.dt))

        if mode == "multi":
            return self._rollout_multi(
                x0,
                n_traj,
                steps_per_traj,
                start_points=start_points,
                init_noise_std=init_noise_std,
                action_noise_std=action_noise_std,
            )

        if mode == "long":
            return self._rollout_long(
                x0,
                n_traj,
                steps_per_traj,
                action_noise_std=action_noise_std,
            )

        raise ValueError("mode must be 'multi' or 'long'")

    # ---------------------- helpers -----------------------------------
    def _rollout_multi(
        self,
        x0: np.ndarray,
        n_traj: int,
        steps_per_traj: int,
        start_points: Optional[Sequence[np.ndarray]],
        init_noise_std: float,
        action_noise_std: float,
    ):
        trajectories = []
        for idx in tqdm(range(n_traj)):
            self.policy.randomize(seed=idx)
            if start_points is not None:
                x = np.copy(start_points[idx]).astype(np.float32)
            else:
                noise = np.random.normal(0.0, init_noise_std, size=x0.shape)
                x = (x0 + noise).astype(np.float32)

            traj = []
            for k in range(steps_per_traj):
                t = k * self.dt
                u = self.policy.act(x, t)
                if action_noise_std > 0.0:
                    u = u + np.random.normal(0.0, action_noise_std, size=u.shape)
                traj.append((np.copy(x), np.copy(u), t))
                x = self.dyn.step(x, u, self.dt)
            trajectories.append(traj)
        return trajectories

    def _rollout_long(
        self,
        x0: np.ndarray,
        n_traj: int,
        steps_per_traj: int,
        action_noise_std: float,
    ):
        total_steps = n_traj * steps_per_traj
        big_traj: list[tuple[np.ndarray, np.ndarray, float]] = []

        x = np.copy(x0).astype(np.float32)
        for k in tqdm(range(total_steps)):
            t = k * self.dt
            u = self.policy.act(x, t)
            if action_noise_std > 0.0:
                u = u + np.random.normal(0.0, action_noise_std, size=u.shape)
            big_traj.append((np.copy(x), np.copy(u), t))
            x = self.dyn.step(x, u, self.dt)

        # Slice into consecutive, non-overlapping windows
        trajectories = [
            big_traj[i * steps_per_traj : (i + 1) * steps_per_traj]
            for i in range(n_traj)
        ]
        return trajectories

    def extract_obs_time_action(self, trajectories):
        obs_list, time_list, act_list = [], [], []

        for traj in trajectories:
            traj_len = len(traj)
            # skip trajectories that are too short
            if traj_len < self.seq_length:
                continue

            # select a random window of length seq_length
            if traj_len > self.seq_length:
                start = random.randint(0, traj_len - self.seq_length)
                window = traj[start : start + self.seq_length]
            else:
                window = traj

            traj_obs, traj_time, traj_act = [], [], []
            for state, action, time_val in window:
                if isinstance(state, torch.Tensor):
                    state = state.cpu().numpy()
                if isinstance(action, torch.Tensor):
                    action = action.cpu().numpy()
                traj_obs.append(state)
                traj_time.append(time_val)
                traj_act.append(action)

            obs_list.append(np.array(traj_obs, dtype=np.float32))
            time_list.append(np.array(traj_time, dtype=np.float32))
            act_list.append(np.array(traj_act, dtype=np.float32))

        return (
            np.array(obs_list, dtype=object),
            np.array(time_list, dtype=object),
            np.array(act_list, dtype=object),
        )
