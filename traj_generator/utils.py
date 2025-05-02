import numpy as np
import plotly.graph_objects as go

def convert_trajs_with_action(trajs):

    n_traj = len(trajs)
    traj_length = len(trajs[0])
    n_dim = trajs[0][0][0].shape[0]
    n_action = trajs[0][0][1].shape[0]

    positions = np.zeros((n_traj, traj_length, n_dim), dtype=np.float32)
    actions = np.zeros((n_traj, traj_length, n_action), dtype=np.float32)
    times = np.zeros((n_traj, traj_length), dtype=np.float32)

    for i, traj in enumerate(trajs):
        for t, (pos, act, time) in enumerate(traj):
            positions[i, t] = pos
            actions[i, t] = act
            times[i, t] = time

    return positions, actions, times

def plot_trajectories(positions, actions, times):
    fig = go.Figure()
    N = 5
    for i in range(N):
        fig.add_trace(go.Scatter3d(
            x=positions[i, :, 0],
            y=positions[i, :, 1],
            z=positions[i, :, 2],
            mode='lines',
            line=dict(width=2),
            name=f'Traj {i}',
            opacity=0.8  # Optional: makes overlapping trajectories clearer
        ))

    fig.update_layout(
        title=f'3D Lorenz Trajectories (First {N})',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        legend=dict(itemsizing='constant'),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.show()
    fig = go.Figure()

def run(gen, mode, n_traj, length, dyn_name, ctrl_name):
    if mode == 'multi':
        starts = [np.random.randn(3).astype(np.float32) for _ in range(n_traj)]
        trajs = gen.rollout(
            x0=np.zeros(3, dtype=np.float32),
            n_traj=n_traj,
            traj_length=length,
            mode="multi",
            start_points=starts,
        )
    elif mode == 'long':
        trajs = gen.rollout(
        x0=np.zeros(3, dtype=np.float32),
        n_traj=n_traj,
        traj_length=length,
        mode="long",
    )
    else:
        RuntimeError('Mode not implemented')
    
    positions, actions, times = convert_trajs_with_action(trajs)
    plot_trajectories(positions, actions, times)
    np.savez(
                f"trajectories/{dyn_name}_{ctrl_name}.npz",
                observations=positions,
                times=times,
                actions=actions
            )
    
    