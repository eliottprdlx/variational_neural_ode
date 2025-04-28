import torch
import torch.nn as nn
from torchdiffeq import odeint as odeint


class ODEFunc(nn.Module):
    def __init__(self, ode_func_net, nonlinear_func = None):
        super(ODEFunc, self).__init__()
        self.ode_func_net = ode_func_net
        self.nonlinear_func = nonlinear_func

    def forward(self, t, z):
        out = self.ode_func_net(z)
        return self.nonlinear_func(out) if self.nonlinear_func else out


class ControlledODEFunc(ODEFunc):
    def __init__(self, ode_func_net, nonlinear_func = None):
        super(ControlledODEFunc, self).__init__(ode_func_net, nonlinear_func)
        self.u: torch.Tensor    # shape (batch, T, u_dim)
        self.times: torch.Tensor  # shape (T,)

    def forward(self, t, z):
        idx = torch.argmin(torch.abs(self.times - t)).item()
        u_t = self.u[:, idx, :]  # (batch, u_dim)
        z_and_u = torch.cat([z, u_t], dim=-1)
        in_feats = self.ode_func_net[0].in_features
        out = self.ode_func_net(z_and_u)
        return self.nonlinear_func(out) if self.nonlinear_func else out


class DiffEqSolver(nn.Module):
    def __init__(self, ode_func, method = 'dopri5', rtol = None, atol = None):
        super(DiffEqSolver, self).__init__()
        self.ode_func = ode_func
        self.method = method
        self.rtol = rtol
        self.atol = atol

    def forward(self, z0, t, u = None,method = None,rtol = None,atol = None):
        method = method or self.method
        rtol   = rtol   or self.rtol
        atol   = atol   or self.atol
        t = t.squeeze(-1)
        # t: (B,T) or (T,)
        if t.ndim == 2:
            # assume every row t[i] is the same
            # you could also sanity-check with torch.allclose
            t = t[0]

        if u is not None:
            # u: (B,T,u_dim)
            self.ode_func.u     = u
            self.ode_func.times = t

        # one batched call: z0 is (B,latent), t is (T,)
        # returns (T, B, latent)
        z = odeint(self.ode_func, z0, t, method=method)
        return z