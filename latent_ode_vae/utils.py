import torch
import torch.nn as nn
import torch.nn.functional as F
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
import numpy as np
from tqdm import tqdm


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


def train(model, dataset, sub_length, num_batches, batch_size, num_epochs, encoder_type):

    opt = torch.optim.Adam(model.parameters(), lr=model.learning_rate)
    clip = 1.0
    total_hist, recon_hist, kl_hist = [], [], []

    for epoch in range(num_epochs):
        model.train()
        ep_tot = ep_rec = ep_kl = 0.0

        for _ in tqdm(range(num_batches), desc=f"Epoch {epoch}"):
            obs, t, act = dataset.sample_subsequences(sub_length, batch_size)
            obs, t, act = obs.to(model.device), t.to(model.device), act.to(model.device)

            x_hat, mu, logvar, _ = model(obs, t, act)
            tot, rec, kl = model.loss_function(x_hat, obs, mu, logvar, epoch)
            tot = rec + kl
            baseline_mse = ((obs - obs.mean(dim=(0, 1), keepdim=True)) ** 2).sum()/obs.size(0)

            opt.zero_grad()
            tot.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()

            ep_tot += tot.item(); ep_rec += rec.item(); ep_kl += kl.item()

        total_hist.append(ep_tot / num_batches)
        recon_hist.append(ep_rec / num_batches)
        kl_hist.append(ep_kl / num_batches)
        print(f"loss {total_hist[-1]:.4f}  "
        f"recon {recon_hist[-1]:.4f}  KL {kl_hist[-1]:.4f}  "
        f"baseline mse {baseline_mse: .4f}")
        if epoch % 20 == 0:
            _plot_reconstruction(
                    obs[0],
                    x_hat[0],
                    act[0],
                    epoch,
                )
    _plot_losses(total_hist, recon_hist, kl_hist, encoder_type)


def train_with_length_scheduler(
    model,
    dataset,
    max_sub_length,
    num_batches,
    batch_size,
    num_epochs,
    encoder_type,
    min_sub_length=10,
    length_step=5,
    epoch_step=20
):
    opt = torch.optim.Adam(model.parameters(), lr=model.learning_rate)
    clip = 1.0
    total_hist, recon_hist, kl_hist = [], [], []

    for epoch in range(num_epochs):
        model.train()
        ep_tot = ep_rec = ep_kl = 0.0

        sub_length = int(min_sub_length + length_step * (epoch // epoch_step))
        sub_length = min(sub_length, max_sub_length)

        for _ in tqdm(range(num_batches), desc=f"Epoch {epoch} | sub_length={sub_length}"):
            obs, t, act = dataset.sample_subsequences(sub_length, batch_size)
            obs, t, act = obs.to(model.device), t.to(model.device), act.to(model.device)

            x_hat, mu, logvar, _ = model(obs, t, act)
            tot, rec, kl = model.loss_function(x_hat, obs, mu, logvar, epoch)

            opt.zero_grad()
            tot.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()

            ep_tot += tot.item()
            ep_rec += rec.item()
            ep_kl += kl.item()
            baseline_mse = ((obs - obs.mean(dim=(0, 1), keepdim=True)) ** 2).sum()/obs.size(0)

        total_hist.append(ep_tot / num_batches)
        recon_hist.append(ep_rec / num_batches)
        kl_hist.append(ep_kl / num_batches)

        print(f"loss {total_hist[-1]:.4f}  "
              f"recon {recon_hist[-1]:.4f}  KL {kl_hist[-1]:.4f}  "
              f"baseline mse {baseline_mse: .4f}")

        if epoch % 20 == 0:
            _plot_reconstruction(obs[0], x_hat[0], act[0], epoch)

    _plot_losses(total_hist, recon_hist, kl_hist, encoder_type)


def _plot_losses(total_losses, recon_losses, kl_losses, encoder_type):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=total_losses, mode='lines', name='Total Loss'))
    fig.add_trace(go.Scatter(y=recon_losses, mode='lines', name='Reconstruction Loss'))
    fig.add_trace(go.Scatter(y=kl_losses, mode='lines', name='KL Divergence Loss'))
    fig.update_layout(title='Losses during Training', xaxis_title='Epochs', yaxis_title='Loss')
    fig.show()
    fig.write_html(f"LatentODE/plots/losses_{encoder_type}.html")


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
