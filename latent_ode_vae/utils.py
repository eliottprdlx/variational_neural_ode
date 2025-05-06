import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import math


def create_mlp(input_dim, hidden_dim, output_dim, num_layers=2, activation='relu'):
    layers = []
    layers.append(nn.Linear(input_dim, hidden_dim))
    
    for _ in range(num_layers - 1):
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        elif activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        
        layers.append(nn.Linear(hidden_dim, hidden_dim))
    
    layers.append(nn.Linear(hidden_dim, output_dim))
    
    return nn.Sequential(*layers)

def pretrain_with_length_warmup(
    model,
    dataset,
    min_sub_length,
    max_sub_length,
    num_batches,
    batch_size,
    length_step=5,
    epoch_step=20,
    masker=None,
):
    num_length_increments = math.ceil((max_sub_length - min_sub_length) / length_step)
    num_epochs = num_length_increments * epoch_step

    print(f"[Pretrain] Automatically setting num_epochs = {num_epochs} to reach sub_length={max_sub_length}")

    opt = torch.optim.Adam([
        {"params": model.encoder.parameters(), "lr": model.encoder_learning_rate},
        {"params": model.ode_func_net.parameters(), "lr": model.ode_learning_rate},
        {"params": model.decoder.parameters(), "lr": model.decoder_learning_rate},
    ])
    clip = 1.0
    total_hist, recon_hist, kl_hist = [], [], []

    for epoch in range(num_epochs):
        model.train()
        ep_tot = ep_rec = ep_kl = 0.0

        sub_length = int(min_sub_length + length_step * (epoch // epoch_step))
        sub_length = min(sub_length, max_sub_length)

        for _ in tqdm(range(num_batches), desc=f"[Pretrain] Epoch {epoch} | sub_length={sub_length}"):
            obs, t, act = dataset.sample_subsequences(sub_length, batch_size)
            obs, t, act = obs.to(model.device), t.to(model.device), act.to(model.device)

            masked_obs = masker(obs) if masker else obs

            x_hat, mu, logvar, _ = model(masked_obs, t, act)
            tot, rec, kl = model.loss_function(x_hat, obs, mu, logvar, epoch)

            opt.zero_grad()
            tot.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()

            ep_tot += tot.item()
            ep_rec += rec.item()
            ep_kl += kl.item()
            baseline_mse = ((obs - obs.mean(dim=(0, 1), keepdim=True)) ** 2).sum() / obs.size(0)

        total_hist.append(ep_tot / num_batches)
        recon_hist.append(ep_rec / num_batches)
        kl_hist.append(ep_kl / num_batches)

        print(f"[Pretrain] loss {total_hist[-1]:.4f}  recon {recon_hist[-1]:.4f}  KL {kl_hist[-1]:.4f}  baseline mse {baseline_mse: .4f}")

        if epoch % 20 == 19:
            _plot_reconstruction_batch(obs, x_hat, act, epoch)

    return total_hist, recon_hist, kl_hist


def train(
    model,
    dataset,
    max_sub_length,
    num_batches,
    batch_size,
    num_epochs,
    encoder_type,
    masker=None,
    scheduler_patience=5,
    scheduler_factor=0.5,
):
    opt = torch.optim.Adam([
        {"params": model.encoder.parameters(), "lr": model.encoder_learning_rate},
        {"params": model.ode_func_net.parameters(), "lr": model.ode_learning_rate},
        {"params": model.decoder.parameters(), "lr": model.decoder_learning_rate},
    ])
    clip = 1.0
    total_hist, recon_hist, kl_hist = [], [], []
    # scheduler = ReduceLROnPlateau(opt, mode='min', factor=scheduler_factor, patience=scheduler_patience)
    scheduler = CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)

    for epoch in range(num_epochs):
        model.train()
        ep_tot = ep_rec = ep_kl = 0.0

        for _ in tqdm(range(num_batches), desc=f"[Train] Epoch {epoch}"):
            obs, t, act = dataset.sample_subsequences(max_sub_length, batch_size)
            obs, t, act = obs.to(model.device), t.to(model.device), act.to(model.device)

            masked_obs = masker(obs) if masker else obs

            x_hat, mu, logvar, _ = model(masked_obs, t, act)
            tot, rec, kl = model.loss_function(x_hat, obs, mu, logvar, epoch)

            opt.zero_grad()
            tot.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()

            scheduler.step(tot.item())

            ep_tot += tot.item()
            ep_rec += rec.item()
            ep_kl += kl.item()
            baseline_mse = ((obs - obs.mean(dim=(0, 1), keepdim=True)) ** 2).sum()/obs.size(0)

        total_hist.append(ep_tot / num_batches)
        recon_hist.append(ep_rec / num_batches)
        kl_hist.append(ep_kl / num_batches)

        print(f"[Train] loss {total_hist[-1]:.4f}  recon {recon_hist[-1]:.4f}  KL {kl_hist[-1]:.4f}  baseline mse {baseline_mse: .4f}")

        if epoch % 20 == 19:
            _plot_reconstruction_batch(obs, x_hat, act, epoch)

    _plot_losses(total_hist, recon_hist, kl_hist, encoder_type)
    return total_hist, recon_hist, kl_hist



def _plot_losses(total_losses, recon_losses, kl_losses, encoder_type):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=total_losses, mode='lines', name='Total Loss'))
    fig.add_trace(go.Scatter(y=recon_losses, mode='lines', name='Reconstruction Loss'))
    fig.add_trace(go.Scatter(y=kl_losses, mode='lines', name='KL Divergence Loss'))
    fig.update_layout(title='Losses during Training', xaxis_title='Epochs', yaxis_title='Loss')
    fig.show()
    fig.write_html(f"latent_ode_vae/plots/losses_{encoder_type}.html")


def _plot_reconstruction(
    obs: torch.Tensor,
    hat: torch.Tensor,
    ctrl: torch.Tensor,
    epoch: int,
):
    obs_np  = obs.detach().cpu().numpy()
    hat_np  = hat.detach().cpu().numpy()
    ctrl_np = ctrl.detach().cpu().numpy()

    T, D = obs_np.shape
    _, C = ctrl_np.shape
    time_axis = np.arange(T)

    titles = [f"state {d}" for d in range(D)] + [f"control {c}" for c in range(C)]
    fig = make_subplots(
        rows=D + C,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=titles,
    )

    for d in range(D):
        fig.add_trace(
            go.Scatter(x=time_axis, y=obs_np[:, d],
                       mode="lines", name=f"state {d} truth"),
            row=d + 1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=time_axis, y=hat_np[:, d],
                       mode="lines", name=f"state {d} recon"),
            row=d + 1, col=1,
        )

    for c in range(C):
        fig.add_trace(
            go.Scatter(x=time_axis, y=ctrl_np[:, c],
                       mode="lines", name=f"control {c}"),
            row=D + c + 1, col=1,
        )

    fig.update_layout(
        height=250 * (D + C),
        width=800,
        title_text=f"Reconstruction + control at epoch {epoch}",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom",
                    y=1.02, xanchor="right", x=1.0),
    )
    fig.update_xaxes(title_text="time step", row=D + C, col=1)
    fig.show()

def _plot_reconstruction_batch(
    obs: torch.Tensor,
    hat: torch.Tensor,
    ctrl: torch.Tensor,
    epoch: int,
    N: int = 3,  # Number of trajectories to plot
):
    obs_np  = obs.detach().cpu().numpy()   # (B, T, D)
    hat_np  = hat.detach().cpu().numpy()   # (B, T, D)
    ctrl_np = ctrl.detach().cpu().numpy()  # (B, T, C)

    B, T, D = obs_np.shape
    _, _, C = ctrl_np.shape
    time_axis = np.arange(T)

    titles = [f"state {d}" for d in range(D)] + [f"control {c}" for c in range(C)]
    fig = make_subplots(
        rows=D + C,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=titles,
    )

    # Create a color map for each (b, d)
    colormap = plt.get_cmap("tab10")

    for b in range(min(N, B)):
        for d in range(D):
            color = colormap(b)  # cycle through colors
            rgba = f"rgba({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)}, 1.0)"

            fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=obs_np[b, :, d],
                    mode="lines",
                    name=f"truth b{b} d{d}",
                    line=dict(dash="solid", color=rgba),
                    showlegend=False,
                ),
                row=d + 1, col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=hat_np[b, :, d],
                    mode="lines",
                    name=f"recon b{b} d{d}",
                    line=dict(dash="dash", color=rgba),
                    showlegend=False,
                ),
                row=d + 1, col=1,
            )

        for c in range(C):
            fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=ctrl_np[b, :, c],
                    mode="lines",
                    name=f"control b{b} c{c}",
                    showlegend=False,
                ),
                row=D + c + 1, col=1,
            )

    fig.update_layout(
        height=250 * (D + C),
        width=800,
        title_text=f"Reconstruction + control at epoch {epoch} ({N} trajectories)",
        showlegend=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )
    fig.update_xaxes(title_text="time step", row=D + C, col=1)
    fig.show()

def plot_imagined_trajectories(
    model,
    dataset,
    sub_length,
    num_samples: int = 5,
    t: torch.Tensor = None,
    u: torch.Tensor = None,
    method: str = None,
    rtol: float = None,
    atol: float = None,
):

    model.eval()
    with torch.no_grad():
        _, t_, u_ = dataset.sample_subsequences(sub_length, num_samples)
        if t is None:
            t = t_.to(model.device)
        if u is None:
            u = u_.to(model.device)

        x_recon, z = model.sample(num_samples, t, u, method=method, rtol=rtol, atol=atol)
        x_recon = x_recon.cpu().numpy()

        fig = go.Figure()

        for i in range(num_samples):
            # Ensure the trajectory has at least 3 dimensions
            assert x_recon.shape[2] >= 3, "Need at least 3D output per timestep to plot 3D trajectories"

            traj = x_recon[i]  # shape: (time_steps, data_dim)
            fig.add_trace(go.Scatter3d(
                x=traj[:, 0],  # x-dim
                y=traj[:, 1],  # y-dim
                z=traj[:, 2],  # z-dim
                mode='lines',
                name=f'Trajectory {i}'
            ))

        fig.update_layout(
            title='3D Trajectories of Imagined Dynamics',
            scene=dict(
                xaxis_title='Dim 1',
                yaxis_title='Dim 2',
                zaxis_title='Dim 3'
            )
        )

        fig.show()