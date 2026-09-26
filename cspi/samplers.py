"""CS-Pi and the gradient / sampling baselines, written once for every benchmark.

A sampler sees one `Problem`: a frozen flow prior with its conditioning bound,
plus the constraint expressed in the prior's normalized space.

    problem.shape            (B, H, d) sample shape
    problem.velocity(x, t)   v_theta(x, t | obs); every call is counted in problem.nfe
    problem.violation(x)     (B,) constraint violation, 0 = feasible
    problem.project(x, final)  the benchmark's projection operator P onto the constraint set
                             (final=True: the terminal fallback, run with a larger iteration budget)
    problem.fallback(x)      terminal fallback of CS-Pi; defaults to project(x, final=True)
    problem.cost(x)          (B,) differentiable cost used by guidance / Trust Sampling
    problem.grad_mask        optional (1, H, d) mask of coordinates gradient methods may move

All samplers return (x, info) with x in normalized space; info["nfe"] is the
number of network evaluations spent on this call.
"""
import math

import torch

from cspi.flow import ode_solve


class Problem:
    """Base class; benchmarks override violation / project / cost."""

    grad_mask = None

    def __init__(self, model, obs_n, shape):
        self.model, self.obs_n, self.shape = model, obs_n, tuple(shape)
        self.nfe = 0

    def velocity(self, x, t):
        self.nfe += 1
        return self.model(x, torch.full((x.shape[0],), t, device=x.device, dtype=x.dtype), self.obs_n)

    def noise(self):
        return torch.randn(self.shape, device=self.obs_n.device)

    def violation(self, x):
        raise NotImplementedError

    def project(self, x, final=False):
        raise NotImplementedError

    def cost(self, x):
        raise NotImplementedError

    def fallback(self, x):
        return self.project(x, final=True)


def _grid_steps(step_size):
    return max(1, round(1.0 / step_size))


# ---------------------------------------------------------------------------
# CS-Pi (Algorithm 1)
# ---------------------------------------------------------------------------

@torch.no_grad()
def cspi(p, k, k_max, tau, step_size=0.1, terminal_projection=True,
         n_schedule="adaptive", fresh_noise=True, n_threshold=None, patience=0, patience_eps_rel=0.05):
    """Constraint-satisfying sampling by projection and partial re-denoising.

        x   <- ODE(eps, 0 -> 1)
        repeat while max_b violation(x) > tau and at most k_max times:
            g     <- P(x)                                   project onto the constraint set
            delta <- max_b ||g - x||                         correction size (batch max)
            n     <- k / (k + delta)                         rollback depth: big fix -> deep rollback
            x_n   <- n g + (1 - n) eps'                      renoise the projection to flow time n
            x     <- ODE(x_n, n -> 1)                        re-denoise with the prior
        if still violating: x <- fallback(x)                 terminal projection (= P for hard constraints)

    Re-denoising uses the same Euler grid as the first solve: ceil((1 - n) / step_size) steps.

    Ablation knobs (tab:earlystop, tab:schedule-ablation):
      n_schedule   "adaptive" (k / (k + delta)) or a constant float n
      fresh_noise  False renoises with zero noise, x_n = n g
      n_threshold  stop once n exceeds this value
      patience     stop when n has closed < patience_eps_rel of its gap to 1 over the last `patience` iterations
    """
    nfe0 = p.nfe
    x = ode_solve(p.velocity, p.noise(), 0.0, 1.0, _grid_steps(step_size))
    viol = p.violation(x)
    n_hist, delta_hist, viol_hist = [], [], [float(viol.max())]
    it, stopped_early = 0, False
    while float(viol.max()) > tau and it < k_max:
        it += 1
        g = p.project(x)
        delta = float((g - x).reshape(x.shape[0], -1).norm(dim=1).max())
        n = k / (k + delta + 1e-8) if n_schedule == "adaptive" else float(n_schedule)
        n_hist.append(n)
        delta_hist.append(delta)
        if n_threshold is not None and n > n_threshold:
            stopped_early = True
            break
        if patience > 0 and len(n_hist) > patience:
            n_prev = n_hist[-1 - patience]
            if (n - n_prev) / max(1.0 - n_prev, 1e-6) < patience_eps_rel:
                stopped_early = True
                break
        x_n = g * n + (torch.randn_like(x) * (1.0 - n) if fresh_noise else 0.0)
        x = ode_solve(p.velocity, x_n, n, 1.0, max(1, math.ceil((1.0 - n) / step_size)))
        viol = p.violation(x)
        viol_hist.append(float(viol.max()))
    fallback = terminal_projection and float(viol.max()) > tau
    if fallback:
        x = p.fallback(x)
    return x, dict(nfe=p.nfe - nfe0, n_iter=it, n_history=n_hist, delta_history=delta_hist,
                   violation_history=viol_hist, fallback=fallback, max_iter_reached=it >= k_max > 0,
                   stopped_early=stopped_early)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

@torch.no_grad()
def unguided(p, step_size=0.1):
    nfe0 = p.nfe
    x = ode_solve(p.velocity, p.noise(), 0.0, 1.0, _grid_steps(step_size))
    return x, dict(nfe=p.nfe - nfe0)


@torch.no_grad()
def projection(p, tau, step_size=0.1):
    """Unguided sample followed by the benchmark projection (CS-Pi with k_max = 0)."""
    return cspi(p, k=1.0, k_max=0, tau=tau, step_size=step_size)


@torch.no_grad()
def rejection(p, budget, tau, step_size=0.1):
    """Draw up to `budget` prior samples per batch element; keep the first feasible one,
    otherwise the least-violating one."""
    nfe0 = p.nfe
    best, best_v, done = None, None, None
    for _ in range(budget):
        x = ode_solve(p.velocity, p.noise(), 0.0, 1.0, _grid_steps(step_size))
        v = p.violation(x)
        if best is None:
            best, best_v, done = x.clone(), v.clone(), v <= tau
        else:
            take = ~done & ((v <= tau) | (v < best_v))
            best = torch.where(take.view(-1, *[1] * (x.dim() - 1)), x, best)
            best_v = torch.where(take, v, best_v)
            done = done | (v <= tau)
        if bool(done.all()):
            break
    return best, dict(nfe=p.nfe - nfe0, accepted=done)


def _cost_grad(p, x_hat):
    """Gradient of sum_b cost(x_hat) with respect to the (detached) point x_hat."""
    with torch.enable_grad():
        xh = x_hat.detach().requires_grad_(True)
        g, = torch.autograd.grad(p.cost(xh).sum(), xh)
    return g if p.grad_mask is None else g * p.grad_mask


@torch.no_grad()
def guidance(p, alpha, step_size=0.1):
    """Gradient guidance on the Tweedie estimate x1_hat = x + (1 - t) v.

    The velocity is corrected by -alpha * grad cost(x1_hat) at every step except the
    last (where guidance would reduce to post-hoc descent on the finished sample).
    """
    nfe0, n = p.nfe, _grid_steps(step_size)
    x, dt = p.noise(), 1.0 / n
    for i in range(n):
        t = i * dt
        v = p.velocity(x, t)
        if alpha > 0 and i < n - 1:
            v = v - alpha * _cost_grad(p, x + (1.0 - t) * v)
        x = x + v * dt
    return x, dict(nfe=p.nfe - nfe0)


@torch.no_grad()
def trust(p, w, j_max, step_size=0.1, stop_tol=1e-6, v_max_mult=1.5, schedule="constant", j_start=0):
    """Trust Sampling (Huang et al., 2024) on a flow prior.

    At every Euler step, take up to J normalized-gradient steps of size w on the cost of
    the Tweedie estimate, differentiating *through* the network. Stop the inner loop
    early when the cost is below stop_tol or when ||v(x*)|| exceeds v_max_mult times the
    unguided ||v(x)|| (manifold guard). Then take the Euler step from the corrected x*.
    schedule="linear" ramps J from j_start (t=0) to j_max (t=1), randomly rounded.
    """
    nfe0, n = p.nfe, _grid_steps(step_size)
    x, dt = p.noise(), 1.0 / n
    B = x.shape[0]
    for i in range(n):
        t = i * dt
        v_ref = float(p.velocity(x, t).reshape(B, -1).norm(dim=1).max())
        if schedule == "linear":
            e = j_start + (j_max - j_start) * (i / (n - 1) if n > 1 else 1.0)
            J = math.floor(e) + (1 if torch.rand(()).item() < e - math.floor(e) else 0)
        else:
            J = j_max
        xs = x.clone()
        for _ in range(J):
            with torch.enable_grad():
                xs = xs.detach().requires_grad_(True)
                vs = p.velocity(xs, t)
                if float(vs.reshape(B, -1).norm(dim=1).max()) > v_max_mult * v_ref:
                    break
                c = p.cost(xs + (1.0 - t) * vs)
                if float(c.max()) <= stop_tol:
                    break
                g, = torch.autograd.grad(c.sum(), xs)
            if p.grad_mask is not None:
                g = g * p.grad_mask
            xs = xs.detach() - w * g / (g.reshape(B, -1).norm(dim=1).view(B, *[1] * (x.dim() - 1)) + 1e-8)
        x = xs.detach() + p.velocity(xs.detach(), t) * dt
    return x, dict(nfe=p.nfe - nfe0)


SAMPLERS = dict(cspi=cspi, unguided=unguided, projection=projection, rejection=rejection,
                guidance=guidance, trust=trust)
