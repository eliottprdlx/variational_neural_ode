import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CriticNetwork(nn.Module):
    """ critic network for estimating state value function """
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.net(x)


class ActorNetwork(nn.Module):
    """ actor network for estimating action probabilities """
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def forward(self, x):
        logits = self.net(x)
        probs = torch.softmax(logits - logits.max(dim=-1, keepdim=True).values, dim=-1)
        return probs


class ActorCriticAgent:
    """ high-level agent """
    def __init__(self, obs_dim, n_actions, device, gamma=0.99, actor_lr=1e-4, critic_lr=1e-3, lambda_=None):
        self.gamma = gamma
        self.lambda_ = lambda_
        self.device = device

        self.actor = ActorNetwork(obs_dim, n_actions).to(self.device)
        self.critic = CriticNetwork(obs_dim).to(self.device)
        self.entropy_coef = 0.01
        self.entropy_decay = 1

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        probs = self.actor(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def update_batch(self, batch):
        """ train the agent using standard td(0) or forward-view td(lambda) """
        """ batch is a list of tuples (s, a, log_pi(a), r, s', done) """
        states, actions, log_probs, rewards, next_states, dones, times = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device)
        log_probs = torch.stack(log_probs).to(self.device)  # these come from `act()`
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        T = len(batch)  # number of transitions in the batch

        with torch.no_grad():
            values      = self.critic(states).squeeze(1)          # shape [T]
            next_values = self.critic(next_states).squeeze(1)

            if not self.lambda_:
                # TD(0) target
                td_targets = rewards + self.gamma * (1 - dones) * next_values
            else:
            # forward-view TD(lambda)
                lambda_returns = []
                for t in range(T):
                    G_lambda = 0.0
                    # for each possible multi-step return from t
                    g = 0.0
                    gamma_pow = 1.0
                    for n in range(T - t):
                        # accumulate n-step return g = R_t + γ R_{t+1} + … + γ^n R_{t+n}
                        g += gamma_pow * rewards[t+n].item()
                        gamma_pow *= self.gamma

                        # bootstrap from V(s_{t+n+1}) if not terminal
                        if dones[t+n].item() == 0:
                            bootstrap = gamma_pow * values[t+n+1].item()
                        else:
                            bootstrap = 0.0

                        # weight by λ^n
                        G_lambda += (self.lambda_**n) * (g + bootstrap)

                        # stop accumulating if done
                        if dones[t+n].item() == 1:
                            break

                    # scale by (1−λ)
                    G_lambda *= (1.0 - self.lambda_)
                    lambda_returns.append(G_lambda)

                td_targets = torch.tensor(lambda_returns,
                                        dtype=torch.float32,
                                        device=self.device).unsqueeze(1)

            advantages = td_targets - values.unsqueeze(1)
        
        # update critic using the TD(0) or TD(lambda) target
        values_pred = self.critic(states)
        critic_loss = F.mse_loss(values_pred, td_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update actor using the TD(0) or TD(lambda) advantage
        probs = self.actor(states)
        dist = torch.distributions.Categorical(probs)
        entropy = dist.entropy().mean()

        actor_loss = -(log_probs * advantages.detach()).mean() - self.entropy_coef * entropy
        self.entropy_coef *= self.entropy_decay  # optional

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()


