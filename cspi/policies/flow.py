"""Flow-matching policies: any velocity field v(x, t, obs), sampled with explicit Euler steps.

Convention (rectified / OT-CFM interpolant):

    x_t = (1 - t) * eps + t * x_1 ,   eps ~ N(0, I),   t in [0, 1]
    target velocity  v = x_1 - eps

so t = 0 is pure noise and t = 1 is data. `FlowPolicy` wraps a velocity function (an nn.Module or a
plain callable) as a `cspi.policies.Policy`; `load_flow_policy` builds one from a released checkpoint.
"""
import copy
import os

import torch

from cspi.policies.base import Policy, full_steps, partial_steps
from cspi.policies.networks import build_model
from cspi.policies.normalizer import Normalizer


def ode_solve(velocity, x, t0=0.0, t1=1.0, n_steps=10):
    """Explicit Euler for dx/dt = velocity(x, t) from t0 to t1."""
    dt = (t1 - t0) / n_steps
    t = t0
    for _ in range(n_steps):
        x = x + velocity(x, t) * dt
        t += dt
    return x


class FlowPolicy(Policy):
    """A flow-matching policy over (horizon, action_dim) samples.

    velocity_fn(x (B, H, d), t (B,), obs_n (B, obs_dim)) -> (B, H, d), in model space.
    normalizer maps model space to physical units (default: identity).
    """

    def __init__(self, velocity_fn, horizon, action_dim, normalizer=None):
        self.velocity_fn, self.horizon, self.action_dim = velocity_fn, horizon, action_dim
        self.normalizer = normalizer if normalizer is not None else Normalizer.identity()

    def condition(self, obs):
        bound = copy.copy(self)
        bound.obs_n = self.normalizer.normalize_obs(obs)
        bound.shape = (obs.shape[0], self.horizon, self.action_dim)
        bound.nfe = 0
        return bound

    def velocity(self, x, t):
        self.nfe += 1
        return self.velocity_fn(x, torch.full((x.shape[0],), t, device=x.device, dtype=x.dtype), self.obs_n)

    def noise(self):
        return torch.randn(self.shape, device=self.obs_n.device)

    def sample(self, step_size=0.1):
        return ode_solve(self.velocity, self.noise(), 0.0, 1.0, full_steps(step_size))

    def denoise(self, x, t0, step_size=0.1):
        return ode_solve(self.velocity, x, t0, 1.0, partial_steps(t0, step_size))


# ---------------------------------------------------------------------------
# Training objective and checkpoint I/O
# ---------------------------------------------------------------------------

def cfm_loss(model, x1, obs, weight=None):
    """OT-CFM loss on normalized data x1 (B, H, d) and normalized obs (B, obs_dim).

    `weight` (B, H) optionally re-weights timesteps of the chunk (Shelf grasp weighting).
    """
    t = torch.rand(x1.shape[0], device=x1.device)
    eps = torch.randn_like(x1)
    x_t = (1 - t[:, None, None]) * eps + t[:, None, None] * x1
    err = (model(x_t, t, obs) - (x1 - eps)) ** 2
    if weight is None:
        return err.mean()
    w = weight.unsqueeze(-1)
    return (w * err).sum() / (w.sum() * err.shape[-1])


def load_prior(path, device="cpu"):
    """Load a released checkpoint -> (model in eval mode, Normalizer, config dict)."""
    ck = torch.load(path, map_location=device)
    model = build_model(ck["config"])
    model.load_state_dict(ck["model"])
    norm = Normalizer(**ck["normalizer"])
    return model.to(device).eval(), norm.to(device), ck["config"]


def save_prior(path, model, normalizer, config, **extra):
    torch.save({"config": config, "model": model.state_dict(),
                "normalizer": {"kind": normalizer.kind, "x_a": normalizer.x_a.cpu(), "x_b": normalizer.x_b.cpu(),
                               "o_a": normalizer.o_a.cpu(), "o_b": normalizer.o_b.cpu()}, **extra}, path)


def load_flow_policy(checkpoint, device="cpu", assets=None):
    """FlowPolicy from a checkpoint saved by cspi.train: a path, or a name in <assets>/checkpoints/."""
    if not os.path.exists(checkpoint):
        checkpoint = os.path.join(assets or os.environ.get("CSPI_ASSETS", "assets"), "checkpoints", f"{checkpoint}.pt")
    model, norm, _ = load_prior(checkpoint, device)
    return FlowPolicy(model, model.horizon, model.action_dim, norm)
