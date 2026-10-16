"""Adapter for epsilon-prediction diffusion policies (DDPM-trained, sampled with deterministic DDIM).

CS-Pi only needs sample / denoise-from-t / renoise-to-t, so a diffusion model plugs in by mapping
flow time t in [0, 1] (0 = noise, 1 = data) onto its discrete training schedule:

    i(t)        = round((1 - t) * T) - 1     training timestep at flow time t (i = -1 means clean)
    abar(t)     = alphas_cumprod[i(t)]        and abar = 1 when i(t) = -1
    renoise     x_t = sqrt(abar(t)) x1 + sqrt(1 - abar(t)) eps
    denoise     DDIM steps on the flow-time grid t0 -> 1

The rollback depth n = k / (k + delta) of CS-Pi is then a flow time on this map, so k has to be
re-tuned for a diffusion policy (the same k does not give the same signal level as for a flow).
No velocity field is exposed: gradient guidance, Trust Sampling and the CBF baselines are flow-only.
"""
import copy

import torch

from cspi.policies.base import Policy, full_steps, partial_steps
from cspi.policies.normalizer import Normalizer


class DDIMPolicy(Policy):
    """eps_fn(x (B, H, d), timestep (B,) long, obs_n (B, obs_dim)) -> predicted noise (B, H, d).

    alphas_cumprod: (T,) the training schedule, alphas_cumprod[0] ~ 1 (least noisy).
    """

    def __init__(self, eps_fn, alphas_cumprod, horizon, action_dim, normalizer=None):
        self.eps_fn, self.horizon, self.action_dim = eps_fn, horizon, action_dim
        self.abar = torch.as_tensor(alphas_cumprod, dtype=torch.float32)
        self.normalizer = normalizer if normalizer is not None else Normalizer.identity()

    def condition(self, obs):
        bound = copy.copy(self)
        bound.obs_n = self.normalizer.normalize_obs(obs)
        bound.shape = (obs.shape[0], self.horizon, self.action_dim)
        bound.nfe = 0
        return bound

    def _index(self, t):
        T = self.abar.shape[0]
        return min(max(int(round((1.0 - t) * T)) - 1, -1), T - 1)

    def _abar(self, t):
        i = self._index(t)
        return 1.0 if i < 0 else float(self.abar[i])

    def noise(self):
        return torch.randn(self.shape, device=self.obs_n.device)

    def renoise(self, x1, t, eps=None):
        a = self._abar(t)
        return x1 * a ** 0.5 + (eps * (1.0 - a) ** 0.5 if eps is not None else 0.0)

    def denoise(self, x, t0, step_size=0.1):
        n = full_steps(step_size) if t0 == 0.0 else partial_steps(t0, step_size)
        ts = [t0 + (1.0 - t0) * i / n for i in range(n + 1)]
        for ta, tb in zip(ts[:-1], ts[1:]):
            i = self._index(ta)
            if i < 0:
                continue
            a, b = self._abar(ta), self._abar(tb)
            self.nfe += 1
            eps = self.eps_fn(x, torch.full((x.shape[0],), i, device=x.device, dtype=torch.long), self.obs_n)
            x0 = (x - (1.0 - a) ** 0.5 * eps) / a ** 0.5
            x = b ** 0.5 * x0 + (1.0 - b) ** 0.5 * eps
        return x
