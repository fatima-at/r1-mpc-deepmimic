import numpy as np
import torch
import torch.nn as nn


class RunningNorm(nn.Module):
    def __init__(self, dim, clip=10.0, eps=1e-8):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))
        self.clip = clip
        self.eps = eps

    @torch.no_grad()
    def update(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        bm = x.mean(0)
        bv = x.var(0, unbiased=False)
        bc = torch.tensor(float(x.shape[0]))
        delta = bm - self.mean
        tot = self.count + bc
        self.mean += delta * bc / tot
        m_a = self.var * self.count
        m_b = bv * bc
        m2 = m_a + m_b + delta ** 2 * self.count * bc / tot
        self.var.copy_(m2 / tot)
        self.count.copy_(tot)

    def forward(self, x):
        z = (x - self.mean) / torch.sqrt(self.var + self.eps)
        return torch.clamp(z, -self.clip, self.clip)


def mlp(sizes, act=nn.ELU, out_gain=1.0):
    layers = []
    for i in range(len(sizes) - 1):
        lin = nn.Linear(sizes[i], sizes[i + 1])
        gain = out_gain if i == len(sizes) - 2 else np.sqrt(2)
        nn.init.orthogonal_(lin.weight, gain)
        nn.init.zeros_(lin.bias)
        layers.append(lin)
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256, 256), log_std_init=-1.5,
                 log_std_min=-4.0, log_std_max=1.0):
        super().__init__()
        self.norm = RunningNorm(obs_dim)
        self.actor = mlp([obs_dim, *hidden, act_dim], out_gain=0.01)
        self.critic = mlp([obs_dim, *hidden, 1], out_gain=1.0)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(log_std_init)))
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def dist(self, obs_n):
        mu = self.actor(obs_n)
        std = torch.exp(torch.clamp(self.log_std, self.log_std_min,
                                    self.log_std_max))
        return torch.distributions.Normal(mu, std)

    def value(self, obs_n):
        return self.critic(obs_n).squeeze(-1)

    @torch.no_grad()
    def normalize(self, obs):
        return self.norm(torch.as_tensor(obs, dtype=torch.float32))

    @torch.no_grad()
    def act(self, obs, deterministic=False, normalized=False):
        o = (torch.as_tensor(obs, dtype=torch.float32) if normalized
             else self.normalize(obs))
        d = self.dist(o)
        a = d.mean if deterministic else d.sample()
        logp = d.log_prob(a).sum(-1)
        return a.numpy(), logp.numpy(), self.value(o).numpy()

    def evaluate(self, obs_n, act):
        d = self.dist(obs_n)
        logp = d.log_prob(act).sum(-1)
        ent = d.entropy().sum(-1)
        return logp, ent, self.value(obs_n)
