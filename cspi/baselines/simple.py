"""Baselines that only need the black-box policy contract: unguided, projection, rejection."""
import torch

from cspi.core.algorithm import cspi


@torch.no_grad()
def unguided(policy, constraint, step_size=0.1):
    nfe0 = policy.nfe
    x = policy.sample(step_size)
    return x, dict(nfe=policy.nfe - nfe0)


@torch.no_grad()
def projection(policy, constraint, tau, step_size=0.1):
    """Unguided sample followed by the constraint's terminal projection (CS-Pi with k_max = 0)."""
    return cspi(policy, constraint, k=1.0, k_max=0, tau=tau, step_size=step_size)


@torch.no_grad()
def rejection(policy, constraint, budget, tau, step_size=0.1):
    """Draw up to `budget` prior samples per batch element; keep the first feasible one,
    otherwise the least-violating one."""
    nfe0 = policy.nfe
    best, best_v, done = None, None, None
    for _ in range(budget):
        x = policy.sample(step_size)
        v = constraint.violation(x)
        if best is None:
            best, best_v, done = x.clone(), v.clone(), v <= tau
        else:
            take = ~done & ((v <= tau) | (v < best_v))
            best = torch.where(take.view(-1, *[1] * (x.dim() - 1)), x, best)
            best_v = torch.where(take, v, best_v)
            done = done | (v <= tau)
        if bool(done.all()):
            break
    return best, dict(nfe=policy.nfe - nfe0, accepted=done)
