import abc
from typing import Protocol, List, Tuple
import torch

# environment interface
class Env(Protocol):
    @property
    def state_dim(self) -> int: ...
    @property
    def action_dim(self) -> int: ...
    @property
    def dt(self) -> float: ...
    def reset(self) -> torch.Tensor: ...
    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool, dict]: ...
    def sample_trajectory(self) -> torch.Tensor: ...

# dataset for trajectories
class Dataset:
    def __init__(self, max_size: int):
        self.buffer = []
        self.max_size = max_size
    def add_episode(self, traj: List[Tuple]):
        self.buffer.append(traj)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)
    def sample_subsequences(self, length: int, batch_size: int): ...

# dynamics model
class DynamicsModel(abc.ABC, torch.nn.Module):
    @abc.abstractmethod
    def predict(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Predict next state given state and action"""
    def rollout(self, init_states: torch.Tensor, policy: 'Policy', horizon: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Roll out model for horizon under policy"""
        states = [init_states]
        actions = []
        for t in range(horizon):
            a = policy(states[-1])
            next_s = self.predict(states[-1], a)
            states.append(next_s)
            actions.append(a)
        return torch.stack(states, dim=1), torch.stack(actions, dim=1)

# actor
class Policy(abc.ABC, torch.nn.Module):
    @abc.abstractmethod
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return action distribution or deterministic action"""

# critic
class ValueFunction(abc.ABC, torch.nn.Module):
    @abc.abstractmethod
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return scalar value estimate"""

# high-level agent
class ModelBasedAgent:
    def __init__(self,
                 env: Env,
                 dynamics: DynamicsModel,
                 policy: Policy,
                 value_fn: ValueFunction,
                 gamma: float = 0.99,
                 lambda_: float = 0.9):
        self.env = env
        self.dynamics = dynamics
        self.policy = policy
        self.value_fn = value_fn
        self.gamma = gamma
        self.lambda_ = lambda_

    def act(self, state: torch.Tensor) -> torch.Tensor:
        return self.policy(state)

    def imagine_and_update(self, dataset: Dataset, horizon: int, batch_size: int):
        # Sample initial states from dataset
        init_states = ...  # implementation detail
        # Imagined rollout
        states, actions = self.dynamics.rollout(init_states, self.policy, horizon)
        # Compute rewards and bootstrap values
        returns, advantages = self.compute_returns_and_advantages(states, actions)
        # Update policy and value networks
        self.update_policy(states[:,0], actions[:,0], advantages)
        self.update_value(states[:,0], returns)

    def compute_returns_and_advantages(self, states, actions):
        # placeholder for TD(lambda) or n-step return computation
        ...

    def update_policy(self, states: torch.Tensor, actions: torch.Tensor, advantages: torch.Tensor):
        # policy gradient step
        ...

    def update_value(self, states: torch.Tensor, returns: torch.Tensor):
        # regression step for value network
        ...

    def train(self,
              dataset: Dataset,
              num_rounds: int,
              episodes_per_round: int,
              horizon: int):
        for rnd in range(num_rounds):
            # 1) Fit dynamics on dataset
            self.fit_dynamics(dataset)
            # 2) Improve policy via imagination
            for _ in range(episodes_per_round):
                self.imagine_and_update(dataset, horizon, batch_size=64)
            # 3) Evaluate in real env and collect data
            new_traj = self.run_real_episode()
            dataset.add_episode(new_traj)

    def fit_dynamics(self, dataset: Dataset):
        # train dynamics model with MLE or ELBO depending on model
        ...

    def run_real_episode(self) -> List[Tuple]:
        state = self.env.reset()
        traj = []
        done = False
        while not done:
            action = self.act(state)
            next_state, reward, done, info = self.env.step(action)
            traj.append((state, action, reward, next_state, done))
            state = next_state
        return traj

# -- Example usage --
def main():
    # instantiate concrete Env, DynamicsModel, Policy, ValueFunction
    env = ...
    dynamics = ...
    policy = ...
    value_fn = ...

    agent = ModelBasedAgent(env, dynamics, policy, value_fn)
    dataset = Dataset(max_size=100)

    agent.train(dataset, num_rounds=50, episodes_per_round=5, horizon=20)

if __name__ == "__main__":
    main()
