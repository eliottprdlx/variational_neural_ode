import torch
import torch.nn as nn
from torchdiffeq import odeint as odeint
from interpolation import linear_interp, gaussian_interp
from typing import Callable, Union, Optional, Dict


class ODEFunc(nn.Module):
    def __init__(self, 
                 ode_func_net : nn.Module, 
                 nonlinear_func : Optional[nn.Module] = None):
        super(ODEFunc, self).__init__()
        self.ode_func_net = ode_func_net
        self.nonlinear_func = nonlinear_func

    def forward(self, t: torch.Tensor, z: torch.Tensor):
        out = self.ode_func_net(z)
        return self.nonlinear_func(out) if self.nonlinear_func else out

class ControlledODEFunc(nn.Module):
    def __init__(self,
                 ode_func_net: nn.Module,
                 nonlinear_func: Optional[nn.Module] = None,
                 *,
                 interp: str = 'linear',
                 interp_kwargs: Optional[dict] = None):
        super().__init__()
        self.ode_func_net = ode_func_net
        self.nonlinear_func = nonlinear_func
        self._interp = interp
        if interp == 'linear':
            self._interp_fn = linear_interp
        elif interp == 'gaussian':
            self._interp_fn = gaussian_interp
        else:
            ValueError(f"Unsupported interpolation method: {interp}")
        self._interp_kwargs = interp_kwargs or {}

        self.register_buffer("times", torch.empty(0))   # (T,)
        self.register_buffer("u",     torch.empty(0))   # (B,T,u_dim)

    def _u_at(self, t: torch.Tensor) -> torch.Tensor:
        return self._interp_fn(self.times, self.u, t, **self._interp_kwargs)

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        u_t = self._u_at(t)                                        # (B,u_dim)
        out = self.ode_func_net(torch.cat([z, u_t], dim=-1))
        return self.nonlinear_func(out) if self.nonlinear_func else out


class DiffEqSolver(nn.Module):
    def __init__(self,
                 ode_func: ControlledODEFunc,
                 *,
                 method: str = "dopri5",
                 rtol:   float = 1e-5,
                 atol:   float = 1e-5):
        super().__init__()
        self.ode_func = ode_func
        self.method, self.rtol, self.atol = method, rtol, atol

    def forward(self,
                z0: torch.Tensor,        # (B, latent)
                t:  torch.Tensor,        # (T,)  OR (B,T) OR (B,T,1)
                u:  Optional[torch.Tensor] = None,        # (B,T,u_dim)
                *,
                method: Optional[str] = None,
                rtol:   Optional[float] = None,
                atol:   Optional[float] = None):
        
        if t.ndim == 3:                           # (B,T,1)
            t = t[0, :, 0]
        elif t.ndim == 2:                         # (B,T)
            t = t[0]
        t = t.to(z0.device)

        if u is not None:
            assert u.shape[1] == t.numel(), "`u` and `t` lengths differ"
            self.ode_func.u     = u.to(z0.device)
            self.ode_func.times = t

        z = odeint(
                self.ode_func,
                z0,
                t,
                method = method or self.method,
                rtol   = rtol   or self.rtol,
                atol   = atol   or self.atol,
        )                              # (T, B, latent)
        return z