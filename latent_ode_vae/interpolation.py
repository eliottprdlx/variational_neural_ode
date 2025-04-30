import torch

def linear_interp(times: torch.Tensor,
                  controls: torch.Tensor,
                  t: torch.Tensor) -> torch.Tensor:
    t = t.squeeze()
    idx_hi = torch.searchsorted(times, t).clamp(1, times.numel() - 1)
    idx_lo = idx_hi - 1
    t0, t1 = times[idx_lo], times[idx_hi]
    u0, u1 = controls[:, idx_lo, :], controls[:, idx_hi, :]
    w = ((t - t0) / (t1 - t0)).view(1, 1)
    return (1 - w) * u0 + w * u1


def gaussian_interp(times: torch.Tensor,
                    controls: torch.Tensor,
                    t: torch.Tensor,
                    *,
                    sigma: float = 1.0,
                    window: int = 3) -> torch.Tensor:
    t = t.squeeze()
    dt = t - times
    w = torch.exp(-0.5 * (dt / sigma) ** 2)
    if window is not None:
        centre = torch.searchsorted(times, t).item()
        lo = max(0, centre - window)
        hi = min(times.numel(), centre + window + 1)
        w, controls = w[lo:hi], controls[:, lo:hi, :]
    w = w / w.sum()
    return (controls * w.view(-1, 1)).sum(dim=1)