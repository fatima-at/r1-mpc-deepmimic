import time

import numpy as np
import torch
import torch.nn as nn

from deepmimic.policy import ActorCritic
from deepmimic.vecenv import TERM_KEYS


DEFAULT_PPO = {
    "steps_per_env": 64,
    "epochs": 5,
    "minibatches": 4,
    "gamma": 0.99,
    "lam": 0.95,
    "clip": 0.2,
    "value_clip": None,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "lr": 3.0e-4,
    "max_grad_norm": 1.0,
    "target_kl": 0.01,
    "adaptive_lr": True,
    "lr_min": 1.0e-5,
    "lr_max": 1.0e-2,
    "hidden": (256, 256),
    "log_std_init": -2.5,
}


class PPO:
    def __init__(self, vec, cfg=None, seed=0, device="cpu"):
        self.vec = vec
        self.cfg = dict(DEFAULT_PPO)
        if cfg:
            self.cfg.update(cfg)
        torch.manual_seed(seed)
        self.device = torch.device(device)
        self.ac = ActorCritic(vec.obs_dim, vec.act_dim,
                              hidden=tuple(self.cfg["hidden"]),
                              log_std_init=self.cfg["log_std_init"]).to(self.device)
        self.opt = torch.optim.Adam(self.ac.parameters(), lr=self.cfg["lr"], eps=1e-5)
        self.obs = vec.reset()
        self.ac.norm.update(self.obs)
        self.total_steps = 0
        self.ep_ret = np.zeros(vec.n_envs)
        self.ep_len = np.zeros(vec.n_envs, dtype=np.int64)
        self.done_rets = []
        self.done_lens = []
        self.done_falls = []

    def collect(self):
        T, N = self.cfg["steps_per_env"], self.vec.n_envs
        O, A = self.vec.obs_dim, self.vec.act_dim
        b_obs = np.zeros((T, N, O), dtype=np.float32)
        b_raw = np.zeros((T, N, O), dtype=np.float32)
        b_act = np.zeros((T, N, A), dtype=np.float32)
        b_logp = np.zeros((T, N), dtype=np.float32)
        b_val = np.zeros((T, N), dtype=np.float32)
        b_rew = np.zeros((T, N), dtype=np.float32)
        b_nextval = np.zeros((T, N), dtype=np.float32)
        b_done = np.zeros((T, N), dtype=np.float32)
        term_acc = np.zeros(len(TERM_KEYS))
        n_acc = 0

        for t in range(T):
            obs_n = self.ac.normalize(self.obs).numpy()
            act, logp, val = self.ac.act(obs_n, normalized=True)
            obs2, rew, term, trunc, finals, terms, lens = self.vec.step(act)

            b_raw[t] = self.obs
            b_obs[t] = obs_n
            b_act[t] = act
            b_logp[t] = logp
            b_val[t] = val
            b_rew[t] = rew
            done = term | trunc
            b_done[t] = done.astype(np.float32)
            term_acc += terms.mean(axis=0)
            n_acc += 1

            nv = np.zeros(N, dtype=np.float32)
            need = [i for i in range(N) if trunc[i] and not term[i]]
            if need:
                fo = np.array([finals[i] for i in need], dtype=np.float32)
                with torch.no_grad():
                    nv[need] = self.ac.value(self.ac.normalize(fo)).numpy()
            b_nextval[t] = nv

            self.ep_ret += rew
            self.ep_len += 1
            for i in range(N):
                if done[i]:
                    self.done_rets.append(float(self.ep_ret[i]))
                    self.done_lens.append(int(self.ep_len[i]))
                    self.done_falls.append(1.0 if term[i] else 0.0)
                    self.ep_ret[i] = 0.0
                    self.ep_len[i] = 0

            self.obs = obs2
            self.total_steps += N

        with torch.no_grad():
            last_val = self.ac.value(self.ac.normalize(self.obs)).numpy()

        adv = np.zeros((T, N), dtype=np.float32)
        gamma, lam = self.cfg["gamma"], self.cfg["lam"]
        gae = np.zeros(N, dtype=np.float32)
        for t in reversed(range(T)):
            if t == T - 1:
                boot = np.where(b_done[t] > 0.5, b_nextval[t], last_val)
            else:
                boot = np.where(b_done[t] > 0.5, b_nextval[t], b_val[t + 1])
            delta = b_rew[t] + gamma * boot - b_val[t]
            gae = delta + gamma * lam * (1.0 - b_done[t]) * gae
            adv[t] = gae
        ret = adv + b_val

        self.ac.norm.update(b_raw.reshape(-1, O))
        stats = {k: v / max(n_acc, 1) for k, v in zip(TERM_KEYS, term_acc)}
        vy = ret.reshape(-1)
        vp = b_val.reshape(-1)
        stats["ev"] = float(1.0 - np.var(vy - vp) / max(np.var(vy), 1e-9))
        stats["ret_mean"] = float(np.mean(vy))
        stats["ret_std"] = float(np.std(vy))
        return (b_obs.reshape(-1, O), b_act.reshape(-1, A), b_logp.reshape(-1),
                adv.reshape(-1), ret.reshape(-1), b_val.reshape(-1)), stats

    def update(self, batch):
        obs, act, logp_old, adv, ret, val_old = [
            torch.as_tensor(x, dtype=torch.float32) for x in batch]
        n = obs.shape[0]
        mb = max(1, n // self.cfg["minibatches"])
        idx = np.arange(n)
        c = self.cfg
        losses = {"policy": [], "value": [], "entropy": [], "kl": [], "clipfrac": []}
        stop = False

        for _ in range(c["epochs"]):
            np.random.shuffle(idx)
            epoch_kl = []
            for s in range(0, n, mb):
                j = idx[s:s + mb]
                if len(j) < 2:
                    continue
                a = adv[j]
                a = (a - a.mean()) / (a.std() + 1e-8)

                logp, ent, v = self.ac.evaluate(obs[j], act[j])
                ratio = torch.exp(logp - logp_old[j])
                l1 = ratio * a
                l2 = torch.clamp(ratio, 1 - c["clip"], 1 + c["clip"]) * a
                pi_loss = -torch.min(l1, l2).mean()

                if c["value_clip"] is not None:
                    vc = val_old[j] + torch.clamp(v - val_old[j],
                                                  -c["value_clip"], c["value_clip"])
                    v_loss = 0.5 * torch.max((v - ret[j]) ** 2,
                                             (vc - ret[j]) ** 2).mean()
                else:
                    v_loss = 0.5 * ((v - ret[j]) ** 2).mean()

                ent_loss = ent.mean()
                loss = pi_loss + c["vf_coef"] * v_loss - c["ent_coef"] * ent_loss

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), c["max_grad_norm"])
                self.opt.step()

                with torch.no_grad():
                    kl = ((ratio - 1) - (logp - logp_old[j])).mean().item()
                    cf = ((ratio - 1).abs() > c["clip"]).float().mean().item()
                losses["policy"].append(pi_loss.item())
                losses["value"].append(v_loss.item())
                losses["entropy"].append(ent_loss.item())
                losses["kl"].append(kl)
                losses["clipfrac"].append(cf)
                epoch_kl.append(kl)

            if c["adaptive_lr"] and epoch_kl:
                m = float(np.mean(epoch_kl))
                lr = self.opt.param_groups[0]["lr"]
                if m > 2.0 * c["target_kl"]:
                    lr = max(c["lr_min"], lr / 1.5)
                elif m < 0.5 * c["target_kl"]:
                    lr = min(c["lr_max"], lr * 1.5)
                for g in self.opt.param_groups:
                    g["lr"] = lr
            if (c["target_kl"] is not None and epoch_kl
                    and float(np.mean(epoch_kl)) > 4.0 * c["target_kl"]):
                stop = True
                break
        out = {k: float(np.mean(v)) if v else 0.0 for k, v in losses.items()}
        out["early_stop"] = float(stop)
        out["lr"] = float(self.opt.param_groups[0]["lr"])
        return out

    def learn(self, total_steps, log_every=1, on_log=None):
        t0 = time.perf_counter()
        start_steps = self.total_steps
        it = 0
        while self.total_steps < total_steps:
            batch, rterms = self.collect()
            info = self.update(batch)
            it += 1
            if it % log_every == 0:
                k = min(100, len(self.done_rets))
                rec = {
                    "iter": it,
                    "steps": self.total_steps,
                    "sps": (self.total_steps - start_steps)
                           / max(time.perf_counter() - t0, 1e-9),
                    "ret": float(np.mean(self.done_rets[-k:])) if k else float("nan"),
                    "len": float(np.mean(self.done_lens[-k:])) if k else float("nan"),
                    "fall": float(np.mean(self.done_falls[-k:])) if k else float("nan"),
                    "std": float(torch.exp(self.ac.log_std.detach()).mean()),
                }
                rec.update(rterms)
                rec.update(info)
                if on_log:
                    on_log(rec)
        return self
