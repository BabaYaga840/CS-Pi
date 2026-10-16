"""CS-Pi (Algorithm 1): constraint-satisfying sampling from a frozen policy by projection and
partial re-denoising.

CS-Pi needs only the black-box part of the two contracts:

    policy      sample(step_size), renoise(x1, t, eps), denoise(x, t0, step_size), nfe
                (cspi/policies/base.py; flows and diffusion models both qualify)
    constraint  violation(x), project(x, final), fallback(x)
                (cspi/constraints/base.py)

Every sampler in this repo (CS-Pi and the baselines in cspi/baselines) has the signature

    sampler(policy, constraint, **hyperparameters) -> (x in model space, info dict with info["nfe"])
"""
import torch


@torch.no_grad()
def cspi(policy, constraint, k, k_max, tau, step_size=0.1, terminal_projection=True,
         n_schedule="adaptive", fresh_noise=True, n_threshold=None, patience=0, patience_eps_rel=0.05):
    """
        x   <- sample from the policy                        (t: 0 -> 1)
        repeat while max_b violation(x) > tau and at most k_max times:
            g     <- P(x)                                    project onto the constraint set
            delta <- max_b ||g - x||                          correction size (batch max)
            n     <- k / (k + delta)                          rollback depth: big fix -> deep rollback
            x_n   <- renoise(g, n, eps')                      = n g + (1 - n) eps' for a flow
            x     <- denoise(x_n, n -> 1)                     re-denoise with the prior
        if still violating: x <- fallback(x)                  terminal projection (= P for hard constraints)

    Re-denoising uses the same Euler grid as the first sample: ceil((1 - n) / step_size) steps.

    Ablation knobs (tab:earlystop, tab:schedule-ablation):
      n_schedule   "adaptive" (k / (k + delta)) or a constant float n
      fresh_noise  False renoises with zero noise, x_n = n g
      n_threshold  stop once n exceeds this value
      patience     stop when n has closed < patience_eps_rel of its gap to 1 over the last `patience` iterations
    """
    nfe0 = policy.nfe
    x = policy.sample(step_size)
    viol = constraint.violation(x)
    n_hist, delta_hist, viol_hist = [], [], [float(viol.max())]
    it, stopped_early = 0, False
    while float(viol.max()) > tau and it < k_max:
        it += 1
        g = constraint.project(x)
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
        x_n = policy.renoise(g, n, torch.randn_like(x) if fresh_noise else None)
        x = policy.denoise(x_n, n, step_size)
        viol = constraint.violation(x)
        viol_hist.append(float(viol.max()))
    fallback = terminal_projection and float(viol.max()) > tau
    if fallback:
        x = constraint.fallback(x)
    return x, dict(nfe=policy.nfe - nfe0, n_iter=it, n_history=n_hist, delta_history=delta_hist,
                   violation_history=viol_hist, fallback=fallback, max_iter_reached=it >= k_max > 0,
                   stopped_early=stopped_early)
