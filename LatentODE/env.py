import numpy as np
import gymnasium as gym


def default_action_sampler(action_space, policy=None):
    if policy is None:
        return action_space.sample()
    else:
        return policy.sample_action(action_space)


class EnvWrapper(gym.Wrapper):
    def __init__(self, env_name: str, seq_len: int, dt: float = None,
                 policy=None, seed: int = None):
        self.env_name = env_name
        self.seq_len = seq_len
        self.dt = dt
        self.policy = policy
        self.seed = seed
        self.env = gym.make(env_name)
        if seed is not None:
            self.env.reset(seed=seed)

    def reset(self):
        obs, info = self.env.reset(seed=self.seed)
        self._obs_buf = [obs]
        self._time_buf = [0.0]
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, info

    def sample_trajectory(self):
        obs = self.reset()
        t = 0.0
        obs_dim = obs.shape
        actions = []
        for step in range(1, self.seq_len):
            # choose action
            if self.policy is None:
                action = self.env.action_space.sample()
            else:
                action = self.policy(obs)
            # step
            obs, reward, done, info = self.step(action)
            # record
            if self.dt is not None:
                t += self.dt
            else:
                # if environment provides time in info
                t = info.get('elapsed_steps', step)
            self._obs_buf.append(obs)
            self._time_buf.append(t)
            actions.append(action)
            if done:
                break

        observations = np.stack(self._obs_buf, axis=0)
        times = np.array(self._time_buf, dtype=float)
        return observations, times, actions


class Dataset:
    def __init__(self, env_name: str, seq_len: int, dt: float = None,
                 policy=None, seed: int = None):
        self.env_name = env_name
        self.seq_len = seq_len
        self.dt = dt
        self.policy = policy
        self.seed = seed

    def __iter__(self):
        # each worker should have its own environment instance
        env_wrapper = GymTrajectoryEnv(
            self.env_name, self.seq_len, self.dt, self.policy, self.seed
        )
        while True:
            obs, times, actions = env_wrapper.sample_trajectory()
            yield obs, times, actions