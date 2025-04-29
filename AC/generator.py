import math
import random
import numpy as np
import torch
import gymnasium as gym
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from agent import ActorCriticAgent


def rbf_kernel(time_points, length_scale, variance):
    time_points = time_points[:, None]
    diff = time_points - time_points.T
    return variance * np.exp(-0.5 * (diff / length_scale) ** 2)


def sample_gp_sequence(num_steps, act_dim, length_scale, variance, seed=None):
    rng = np.random.default_rng(seed)
    t = np.arange(num_steps, dtype=np.float32)
    kernel = rbf_kernel(t, length_scale, variance)
    kernel += 1e-6 * np.eye(num_steps)          # jitter for stability
    cholesky = np.linalg.cholesky(kernel)
    normal = rng.standard_normal((num_steps, act_dim), dtype=np.float32)
    return cholesky @ normal                    # (T, act_dim)


def continuous_to_discrete(x, num_classes):
    phi = 0.5 * (1.0 + np.erf(x / math.sqrt(2.0)))        # CDF of N(0,1)
    return np.minimum((phi * num_classes).astype(np.int64), num_classes - 1)


class Generator:
    def __init__(self, env_name="CartPole-v1",device="cuda:0",episodes=500,gamma=0.99,actor_lr=1e-4,critic_lr=1e-3,gae_lambda=None,verbose=False,seq_length=200,
        dt=0.02, use_gp=False, gp_ratio=0.0,gp_kernel=None,gp_seed=None):
        self.env = gym.make(env_name)
        self.env_name = env_name
        self.device = device
        self.episodes = episodes
        self.verbose = verbose
        self.seq_length = seq_length
        self.dt = dt
        self.gae_lambda = gae_lambda
        self.gp_ratio = gp_ratio
        self.gp_kernel = gp_kernel or {"length_scale": 0.5, "variance": 1.0}
        self.gp_seed = gp_seed
        self.use_gp = use_gp

        obs_dim = self.env.observation_space.shape[0]
        self.discrete = isinstance(self.env.action_space, gym.spaces.Discrete)

        if self.discrete:
            self.num_actions = self.env.action_space.n
            act_dim = 1
        else:
            self.act_low = self.env.action_space.low
            self.act_high = self.env.action_space.high
            act_dim = self.env.action_space.shape[0]

        self.agent = ActorCriticAgent(
            obs_dim,
            self.num_actions if self.discrete else act_dim,
            device,
            gamma,
            actor_lr,
            critic_lr,
            gae_lambda,
        )

    def gp_action(self, gp_seq, step_index):
        value = gp_seq[step_index]          # shape (), (d,)
        if self.discrete:
            return int(
                continuous_to_discrete(np.atleast_1d(value), self.num_actions)[0]
            )

        value  = np.atleast_1d(value)       # shape (d,) even when d == 1
        scale  = 0.5 * (self.act_high - self.act_low)
        centre = 0.5 * (self.act_high + self.act_low)
        action = np.clip(value * scale + centre, self.act_low, self.act_high)
        return action.astype(np.float32)    # e.g. array([0.01], dtype=float32)


    def run(self):
        all_rewards, all_trajectories = [], []
        max_steps = self.env.spec.max_episode_steps or 500

        for episode in range(self.episodes):
            render = (episode + 1) % 20 == 0 and self.verbose
            env = gym.make(self.env.spec.id, render_mode="human") if render else self.env
            state, _ = env.reset()
            trajectory, total_reward, done, time_counter = [], 0.0, False, 0.0

            if self.use_gp:
                gp_seq = sample_gp_sequence(
                    max_steps,
                    1 if self.discrete else self.act_low.size,
                    self.gp_kernel["length_scale"],
                    self.gp_kernel["variance"],
                    self.gp_seed,
                )

            step_index = 0
            while not done:
                if self.use_gp:
                    action = self.gp_action(gp_seq, step_index)
                    log_prob = None
                else:
                    action, log_prob = self.agent.act(state)
                
                if not self.discrete:
                    if isinstance(action, torch.Tensor):
                        action = action.detach().cpu().numpy()
                    action = np.atleast_1d(action).astype(np.float32)          # shape (d,)
                    action = np.clip(action, self.act_low, self.act_high)

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                trajectory.append(
                    (state, action, log_prob, reward, next_state, done, time_counter)
                )

                state, total_reward = next_state, total_reward + reward
                time_counter += self.dt
                step_index += 1

            if not self.use_gp:
                self.agent.update_batch(trajectory)

            all_rewards.append(total_reward)
            all_trajectories.append(trajectory)

            mode = "GP" if self.use_gp else "RL"
            print(f"Episode {episode + 1:3d} | {mode} | Reward = {total_reward:.2f}")

            if render:
                env.close()

            if len(all_rewards) > 10 and np.mean(all_rewards[-10:]) >= 500:
                print(f"Environment solved in {episode + 1} episodes!")
                break

        self.plot_learning_curve(all_rewards)
        self.save_trajectories(all_trajectories, mode)

    # ------------------------------- I/O helpers ---------------------------- #
    def plot_learning_curve(self, rewards):
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=rewards, mode="lines+markers", name="Episode reward"))
        if len(rewards) >= 30:
            avg = np.convolve(rewards, np.ones(30) / 30, mode="valid")
            fig.add_trace(go.Scatter(y=avg, mode="lines", name="Moving avg (30 ep)"))
        fig.update_layout(
            title="Reward over Episodes",
            xaxis_title="Episode",
            yaxis_title="Total Reward",
            template="plotly_white",
        )
        fig.show()

    def save_trajectories(self, trajectories, mode):
        observations, times, actions = self.extract_obs_time_action(trajectories)
        suffix_lambda = "_lambda" if self.gae_lambda and not self.use_gp else ""
        np.savez(
            f"AC/trajectories/{mode}_{self.env_name}_trajectories{suffix_lambda}.npz",
            observations=observations,
            times=times,
            actions=actions,
        )

    def extract_obs_time_action(self, trajectories):
        obs_list, time_list, act_list = [], [], []
        for traj in trajectories:
            length = len(traj)
            if length < self.seq_length:
                continue
            start = random.randint(0, length - self.seq_length) if length > self.seq_length else 0
            window = traj[start : start + self.seq_length]
            obs, times, acts = [], [], []
            for state, action, _, _, _, _, t in window:
                obs.append(np.array(state, dtype=np.float32))
                acts.append(np.array(action, dtype=np.float32))
                times.append(t)
            obs_list.append(np.stack(obs))
            act_list.append(np.stack(acts))
            time_list.append(np.array(times))
        return np.array(obs_list, dtype=object), np.array(time_list, dtype=object), np.array(act_list, dtype=object)
