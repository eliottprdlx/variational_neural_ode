import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from agent import ActorCriticAgent
import gymnasium as gym
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
from config import *

class Trainer:
    def __init__(self, env_name="CartPole-v1", device="cuda:0", episodes=500, gamma=0.99, actor_lr=1e-4, critic_lr=1e-3, lambda_=None, verbose=False, seq_length=200, dt=0.02):
        self.env = gym.make(env_name)
        self.env_name = env_name
        obs_dim = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n
        self.agent = ActorCriticAgent(obs_dim, n_actions, device, gamma, actor_lr, critic_lr, lambda_)
        self.lambda_ = lambda_
        self.episodes = episodes
        self.verbose = verbose
        self.seq_length = seq_length
        self.dt = dt

    def run(self):
        all_rewards = []
        all_trajectories = []

        for episode in range(self.episodes):
            render = (episode + 1) % 20 == 0
            if render and self.verbose:
                print(f"\n[Episode {episode + 1}] Rendering...")
                env = gym.make(self.env.spec.id, render_mode="human")
            else:
                env = self.env

            state, _ = env.reset()
            done = False
            total_reward = 0
            trajectory = []  # Collect transitions for one episode
            time_counter = 0.0 

            while not done:
                action, log_prob = self.agent.act(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # Store (s, a, log_pi(a), r, s', done)
                trajectory.append((state, action, log_prob, reward, next_state, done, time_counter))

                state = next_state
                total_reward += reward
                time_counter += self.dt

            # Update from this trajectory
            self.agent.update_batch(trajectory)
            all_trajectories.append(trajectory)
            

            all_rewards.append(total_reward)
            print(f"Episode {episode + 1}: Reward = {total_reward}")

            if render:
                env.close()
            
            # break the loop if moving avg reward is maxed out
            if len(all_rewards) > 10 and np.mean(all_rewards[-10:]) >= 500:
                print(f"Environment solved in {episode + 1} episodes!")
                break

        # Plot rewards
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=all_rewards, mode='lines+markers', name='Episode Reward'))
        # add a moving average line
        window_size = 30
        moving_avg = np.convolve(all_rewards, np.ones(window_size)/window_size, mode='valid')
        fig.add_trace(go.Scatter(y=moving_avg, mode='lines', name='Moving Average'))

        fig.update_layout(
            title='Reward over Episodes',
            xaxis_title='Episode',
            yaxis_title='Total Reward',
            template='plotly_white'
        )
        fig.show()
        obs_np, time_np, act_np = self.extract_obs_time_action(all_trajectories)   
        if self.lambda_:
            fig.write_html(f"AC/plots/{self.env_name}_rewards_lambda.html")
            np.savez(
                f"AC/trajectories/{self.env_name}_trajectories_lambda.npz",
                observations=obs_np,
                times=time_np,
                actions=act_np
            )
        else:
            fig.write_html(f"AC/plots/{self.env_name}_rewards.html")
            np.savez(
                f"AC/trajectories/{self.env_name}_trajectories_lambda.npz",
                observations=obs_np,
                times=time_np,
                actions=act_np
            )
    
    def extract_obs_time_action(self, trajectories):
        obs_list, time_list, act_list = [], [], []

        for traj in trajectories:
            L = len(traj)
            # skip too‐short trajectories
            if L < self.seq_length:
                continue

            # pick a random window of length seq_length
            if L > self.seq_length:
                start = random.randint(0, L - self.seq_length)
                window = traj[start : start + self.seq_length]
            else:
                # exactly equal
                window = traj

            traj_obs, traj_time, traj_act = [], [], []
            for transition in window:
                state, action, log_prob, reward, next_state, done, time = transition

                # convert to numpy if still a tensor
                if isinstance(state, torch.Tensor):
                    state = state.cpu().numpy()
                if isinstance(action, torch.Tensor):
                    action = action.cpu().numpy()

                traj_obs.append(state)
                traj_time.append(time)
                traj_act.append(action)

            obs_list.append(np.array(traj_obs, dtype=np.float32))
            time_list.append(np.array(traj_time, dtype=np.float32))
            act_list.append(np.array(traj_act, dtype=np.float32))

        return (
            np.array(obs_list, dtype=object),
            np.array(time_list, dtype=object),
            np.array(act_list, dtype=object)
        )


    def _plot_trajectory(self, trajectory, episode):
        """Interactive Plotly plot of per-coordinate state trajectory for one episode."""
        states = np.stack([step[0] for step in trajectory], axis=0)   # (T, D)
        times  = np.array([step[6] for step in trajectory])           # (T,)
        T, D   = states.shape

        fig = make_subplots(rows=D, cols=1, shared_xaxes=True, vertical_spacing=0.02)
        for d in range(D):
            fig.add_trace(
                go.Scatter(x=times, y=states[:, d], mode="lines", name=f"state {d}"),
                row=d + 1,
                col=1,
            )
        fig.update_layout(
            height=250 * D,
            width=900,
            title_text=f"State trajectory – episode {episode + 1}",
            showlegend=D == 1,   # legend only when single plot
            xaxis_title="time (s)",
            template="plotly_white",
        )
        fig.update_xaxes(title_text="time (s)", row=D, col=1)
        fig.show()