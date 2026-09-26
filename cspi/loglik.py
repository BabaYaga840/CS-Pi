"""Exact log-likelihood of a sample under the flow prior (continuous normalizing flow).

A flow-matching model defines the ODE dx/dt = v(x, t | obs) with x(0) ~ N(0, I), so

    log p(x1 | obs) = log N(x0; 0, I) - int_0^1 div v(x_t, t | obs) dt,

where x0 is found by integrating backward from x1. The divergence is estimated
with Hutchinson (Rademacher) probes. The paper reports log p / D ("nats/dim"),
averaged over samples; higher means more in-distribution.
"""
import math

import torch


def _velocity_and_divergence(model, x, t, obs, n_probes):
    with torch.enable_grad():
        x = x.detach().requires_grad_(True)
        v = model(x, torch.full((x.shape[0],), t, dtype=x.dtype, device=x.device), obs)
        div = torch.zeros(x.shape[0], device=x.device)
        for _ in range(n_probes):
            e = torch.randint(0, 2, x.shape, dtype=x.dtype, device=x.device) * 2 - 1
            g, = torch.autograd.grad(v, x, grad_outputs=e, retain_graph=True)
            div += (g * e).reshape(x.shape[0], -1).sum(1)
    return v.detach(), div / n_probes


def log_likelihood(model, x1, obs, n_steps=24, n_probes=6, batch=128):
    """log p(x1 | obs) / D for normalized samples x1 (N, H, d) and normalized obs (N, obs_dim)."""
    out = []
    D, dt = x1[0].numel(), 1.0 / n_steps
    for i in range(0, x1.shape[0], batch):
        x, o = x1[i:i + batch].clone(), obs[i:i + batch]
        integral = torch.zeros(x.shape[0], device=x.device)
        for j in reversed(range(n_steps)):
            v, div = _velocity_and_divergence(model, x, (j + 1) / n_steps, o, n_probes)
            x = x - v * dt
            integral = integral + div * dt
        log_p0 = -0.5 * (x.reshape(x.shape[0], -1) ** 2).sum(1) - 0.5 * D * math.log(2 * math.pi)
        out.append((log_p0 - integral) / D)
    return torch.cat(out)
