"""
controls.py
===========

Smooth control policies that can be queried as

    u = policy.act(state, t)
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple, Literal, Sequence, Optional


class ControlPolicy(ABC):
    @abstractmethod
    def act(self, x: np.ndarray, t: float) -> np.ndarray:
        """Return the control value u(t)."""

class RandomSinusControl(ControlPolicy):
    """
    u(t) = sum_{k=1..n_freq} amplitude_k * sin(2 * pi * frequency_k * t + phase_k)

    Amplitudes, frequencies and phases are drawn independently per
    dimension, resulting in a smooth, band-limited signal.
    """

    def __init__(
        self,
        dim: int = 1,
        n_freq: int = 3,
        amp_range: tuple[float, float] = (0.5, 1.0),
        freq_range: tuple[float, float] = (0.2, 1.5),
        seed: Optional[int] = None,
    ):
        rng = np.random.default_rng(seed)
        self.dim = dim
        self.n_freq = n_freq
        self.amplitude = rng.uniform(*amp_range, size=(dim, n_freq))
        self.frequency = rng.uniform(*freq_range, size=(dim, n_freq))
        self.phase = rng.uniform(0.0, 2.0 * np.pi, size=(dim, n_freq))
    
    def randomize(self, seed: Optional[int] = None):
        """Reinitialize the control signal with a new random seed."""
        rng = np.random.default_rng(seed)
        self.amplitude = rng.uniform(*self.amp_range, size=(self.dim, self.n_freq))
        self.frequency = rng.uniform(*self.freq_range, size=(self.dim, self.n_freq))
        self.phase = rng.uniform(0.0, 2.0 * np.pi, size=(self.dim, self.n_freq))

    def act(self, x: np.ndarray, t: float) -> np.ndarray:  # noqa: D401
        angle = 2.0 * np.pi * self.frequency * t + self.phase # shape (dim, n_freq)
        value = np.sum(self.amplitude * np.sin(angle), axis=1)
        return value.astype(np.float32)


class GaussianProcessControl(ControlPolicy):
    """
    A smooth control signal sampled from an RBF Gaussian process.

    Implementation uses random Fourier features.

        u(t) = sqrt(2 * variance / n_features) *
               sum_{k=1..n_features} alpha_k * cos(omega_k * t + bias_k)

    where
        omega_k ~ N(0, 1 / length_scale^2)
        bias_k  ~ Uniform(0, 2 pi)
        alpha_k ~ N(0, 1)
    """

    def __init__(
        self,
        dim: int = 1,
        n_features: int = 1000,
        length_scale: float = 1.0,
        variance: float = 1.0,
        seed: Optional[int] = None,
    ):
        rng = np.random.default_rng(seed)
        self.dim = dim
        self.n = n_features
        self.prefactor = np.sqrt(2.0 * variance / n_features)
        self.length_scale = length_scale

        self.omega = rng.normal(0.0, 1.0 / length_scale, size=(dim, n_features))
        self.bias = rng.uniform(0.0, 2.0 * np.pi, size=(dim, n_features))
        self.alpha = rng.normal(0.0, 1.0, size=(dim, n_features))
    
    def randomize(self, seed: Optional[int] = None):
        """Reinitialize the control signal with a new random seed."""
        rng = np.random.default_rng(seed)
        self.omega = rng.normal(0.0, 1.0 / self.length_scale, size=(self.dim, self.n))
        self.bias = rng.uniform(0.0, 2.0 * np.pi, size=(self.dim, self.n))
        self.alpha = rng.normal(0.0, 1.0, size=(self.dim, self.n))

    def act(self, x: np.ndarray, t: float) -> np.ndarray:  # noqa: D401
        arg = self.omega * t + self.bias
        value = self.prefactor * np.sum(self.alpha * np.cos(arg), axis=1)
        return value.astype(np.float32)
