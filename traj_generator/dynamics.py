from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp


class Dynamics(ABC):
    @abstractmethod
    def rhs(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Right hand side of the differential equation x_dot = rhs(t, x, u)."""

    def step(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """
        Integrate the system forward by dt seconds.

        Parameters
        ----------
        x  : current state (numpy array, shape = (n_dim,))
        u  : control input (numpy array, shape decided by the model)
        dt : integration horizon in seconds
        """
        sol = solve_ivp(
            fun=lambda t, y: self.rhs(t, y, u),
            t_span=(0.0, dt),
            y0=x,
            method="RK45",
            rtol=1e-6,
            atol=1e-9,
            max_step=dt,       # at most one solver output step
        )
        return sol.y[:, -1].astype(np.float32)


# ------------------------------------------------------------------
# Lorenz attractor (no control)
# ------------------------------------------------------------------
class Lorenz(Dynamics):
    """
    Lorenz system with additive control input

        x_dot = sigma * (y - x) + gain[0] * u[0]
        y_dot = x * (rho - z) - y + gain[1] * u[1]
        z_dot = x * y - beta * z + gain[2] * u[2]

    Parameters
    ----------
    sigma, rho, beta : classic Lorenz parameters
    gain             : length-3 vector scaling each control component
                       (default 0,0,0 reproduces the uncontrolled model)
    """

    def __init__(
        self,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8.0 / 3.0,
        gain: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.gain = np.asarray(gain, dtype=np.float32)
    
    @property
    def state_dim(self) -> int:
        return 3

    def rhs(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        # Ensure we have a usable control vector
        if u is None:
            u = np.zeros(3, dtype=np.float32)
        x1, x2, x3 = x
        u1, u2, u3 = u
        g1, g2, g3 = self.gain
        return np.array(
            [
                self.sigma * (x2 - x1) + g1 * u1,
                x1 * (self.rho - x3) - x2 + g2 * u2,
                x1 * x2 - self.beta * x3 + g3 * u3,
            ],
            dtype=np.float32,
        )

class SimplePendulum(Dynamics):
    """
    State  x = [theta, omega]
    Input  u = [torque]

        theta_dot = omega
        omega_dot = -(g / L) * sin(theta) - d * omega + torque / (m L^2)
    """

    def __init__(
        self,
        length: float = 1.0,
        mass: float = 1.0,
        gravity: float = 9.81,
        damping: float = 0.0,
    ):
        self.length = length
        self.mass = mass
        self.gravity = gravity
        self.damping = damping
    
    @property
    def state_dim(self) -> int:
        return 2

    def rhs(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        theta, omega = x
        torque = float(u[0]) if u is not None else 0.0
        theta_dot = omega
        omega_dot = (
            -self.gravity / self.length * np.sin(theta)
            - self.damping * omega
            + torque / (self.mass * self.length ** 2)
        )
        return np.array([theta_dot, omega_dot], dtype=np.float32)


# ------------------------------------------------------------------
# Double pendulum (two independent torques)
# ------------------------------------------------------------------
class DoublePendulum(Dynamics):
    """
    State  x = [theta1, omega1, theta2, omega2]
    Input  u = [torque1, torque2]

    The equations are taken from the standard energy formulation of the
    double planar pendulum with point masses.
    """

    def __init__(
        self,
        mass1: float = 1.0,
        mass2: float = 1.0,
        length1: float = 1.0,
        length2: float = 1.0,
        gravity: float = 9.81,
        damping1: float = 0.0,
        damping2: float = 0.0,
    ):
        self.m1 = mass1
        self.m2 = mass2
        self.L1 = length1
        self.L2 = length2
        self.g = gravity
        self.d1 = damping1
        self.d2 = damping2
    
    @property
    def state_dim(self) -> int:
        return 4

    def rhs(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        theta1, omega1, theta2, omega2 = x
        if u is None:
            torque1, torque2 = 0.0, 0.0
        else:
            torque1, torque2 = float(u[0]), float(u[1])

        delta = theta2 - theta1
        sin_delta = np.sin(delta)
        cos_delta = np.cos(delta)

        m1, m2, L1, L2, g = self.m1, self.m2, self.L1, self.L2, self.g

        denom = (m1 + m2) * L1 - m2 * L1 * cos_delta * cos_delta

        omega1_dot = (
            m2 * L1 * omega1 * omega1 * sin_delta * cos_delta
            + m2 * g * np.sin(theta2) * cos_delta
            + m2 * L2 * omega2 * omega2 * sin_delta
            - (m1 + m2) * g * np.sin(theta1)
        ) / denom + torque1 / ((m1 + m2) * L1 * L1) - self.d1 * omega1

        omega2_dot = (
            -(m1 + m2) * L2 * omega2 * omega2 * sin_delta * cos_delta
            + (m1 + m2) * g * np.sin(theta1) * cos_delta
            - (m1 + m2) * L1 * omega1 * omega1 * sin_delta
            - (m1 + m2) * g * np.sin(theta2)
        ) / (L2 / L1 * denom) + torque2 / (m2 * L2 * L2) - self.d2 * omega2

        return np.array([omega1, omega1_dot, omega2, omega2_dot], dtype=np.float32)
