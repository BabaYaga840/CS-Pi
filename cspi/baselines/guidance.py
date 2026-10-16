"""Gradient guidance and Trust Sampling on a flow policy (need policy.velocity and constraint.cost)."""
import math

import torch

from cspi.policies.base import full_steps


def cost_grad(constraint, x_hat):
    """Gradient of sum_b cost(x_hat) with respect to the (detached) point x_hat, masked by grad_mask."""
    with torch.enable_grad():
        xh = x_hat.detach().requires_grad_(True)
        g, = torch.autograd.grad(constraint.cost(xh).sum(), xh)
    return g if constraint.grad_mask is None else g * constraint.grad_mask


@torch.no_grad()
def guidance(policy, constraint, alpha, step_size=0.1):
    """Gradient guidance on the Tweedie estimate x1_hat = x + (1 - t) v.

    The velocity is corrected by -alpha * grad cost(x1_hat) at every step except the
    last (where guidance would reduce to post-hoc descent on the finished sample).
    """
    nfe0, n = policy.nfe, full_steps(step_size)
    x, dt = policy.noise(), 1.0 / n
    for i in range(n):
        t = i * dt
        v = policy.velocity(x, t)
        if alpha > 0 and i < n - 1:
            v = v - alpha * cost_grad(constraint, x + (1.0 - t) * v)
        x = x + v * dt
    return x, dict(nfe=policy.nfe - nfe0)


@torch.no_grad()
def trust(policy, constraint, w, j_max, step_size=0.1, stop_tol=1e-6, v_max_mult=1.5, schedule="constant",
          j_start=0):
    """Trust Sampling (Huang et al., 2024) on a flow prior.

    At every Euler step, take up to J normalized-gradient steps of size w on the cost of
    the Tweedie estimate, differentiating *through* the network. Stop the inner loop
    early when the cost is below stop_tol or when ||v(x*)|| exceeds v_max_mult times the
    unguided ||v(x)|| (manifold guard). Then take the Euler step from the corrected x*.
    schedule="linear" ramps J from j_start (t=0) to j_max (t=1), randomly rounded.
    """
    nfe0, n = policy.nfe, full_steps(step_size)
    x, dt = policy.noise(), 1.0 / n
    B = x.shape[0]
    grad_mask = constraint.grad_mask
    for i in range(n):
        t = i * dt
        v_ref = float(policy.velocity(x, t).reshape(B, -1).norm(dim=1).max())
        if schedule == "linear":
            e = j_start + (j_max - j_start) * (i / (n - 1) if n > 1 else 1.0)
            J = math.floor(e) + (1 if torch.rand(()).item() < e - math.floor(e) else 0)
        else:
            J = j_max
        xs = x.clone()
        for _ in range(J):
            with torch.enable_grad():
                xs = xs.detach().requires_grad_(True)
                vs = policy.velocity(xs, t)
                if float(vs.reshape(B, -1).norm(dim=1).max()) > v_max_mult * v_ref:
                    break
                c = constraint.cost(xs + (1.0 - t) * vs)
                if float(c.max()) <= stop_tol:
                    break
                g, = torch.autograd.grad(c.sum(), xs)
            if grad_mask is not None:
                g = g * grad_mask
            xs = xs.detach() - w * g / (g.reshape(B, -1).norm(dim=1).view(B, *[1] * (x.dim() - 1)) + 1e-8)
        x = xs.detach() + policy.velocity(xs.detach(), t) * dt
    return x, dict(nfe=policy.nfe - nfe0)
