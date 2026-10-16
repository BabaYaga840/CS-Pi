"""Control-barrier-function baselines on a flow prior: SafeDiffuser and SafeFlowMatcher.

Both replace the velocity v at an Euler step by the closest velocity u that satisfies the
barrier conditions of the constraint, solved as a QP in the normalized model space:

    min_{u, r} ||u - v||^2 + ||r||^2   s.t.   a_i^T u + w_i r_i >= c_i  for every barrier row i

    a_i = s * d b_i / d x   evaluated at x ("ode") or at the Tweedie estimate x + (1 - t) v
                            ("tweedie", then s = 1 - t);
    c_i = -lam (b_i - delta)                          (exponential class-K, SafeDiffuser)
    c_i = -eps sgn(b_i - delta) |b_i - delta|^rho     (finite time, SafeFlowMatcher)

w_i > 0 relaxes a row with a slack variable (a constant weight, or SafeFlowMatcher's schedule
w_t = w_scale (1 - exp(3 (t / t_w - 1))) that vanishes for t >= t_w).

SafeDiffuser    : the QP filter at every one of n_steps Euler steps.
SafeFlowMatcher : T_p plain Euler steps (prediction), then flow time restarts at 0 and T_c
                  correction steps follow the vanishing field alpha (1 - t) v under the filter.

The constraint supplies constraint.barrier_rows(x) -> (b (B, R), A (B, R, n)): barrier values and
their Jacobian with respect to the flattened normalized sample; the policy must expose velocity(x, t).
"""
import math

import numpy as np
import torch

# ---------------------------------------------------------------------------
# QP:  min ||u - v||^2 + ||r||^2  s.t.  A u + w r >= c     (float64, one sample)
# ---------------------------------------------------------------------------

_TOL = 1e-9


def _dual_active_set(G, h, max_iter=200):
    """Lawson-Hanson active set for  min_{lam >= 0}  lam^T G lam / 2 - h^T lam."""
    m = len(h)
    lam, P = np.zeros(m), np.zeros(m, dtype=bool)
    for _ in range(max_iter):
        g = G @ lam - h
        cand = np.where(~P & (g < -1e-12))[0]
        if cand.size == 0:
            break
        P[cand[np.argmin(g[cand])]] = True
        for _ in range(max_iter):
            idx = np.where(P)[0]
            s = np.linalg.lstsq(G[np.ix_(idx, idx)], h[idx], rcond=None)[0]
            if (s > 1e-12).all():
                lam[:] = 0.0
                lam[idx] = s
                break
            li = lam[idx]
            neg = (s <= 1e-12) & (li - s > 1e-12)
            alpha = np.min(li[neg] / (li[neg] - s[neg])) if neg.any() else 0.0
            lam[idx] = np.maximum(li + alpha * (s - li), 0.0)
            newP = lam > 1e-12
            if (newP == P).all():
                newP[idx[np.argmin(s)]] = False
            P = newP
    return lam


def _fista(G, h, lam, iters=4000):
    L = np.linalg.eigvalsh(G).max() if len(h) <= 256 else np.abs(G).sum(1).max()
    y, t, lam = lam.copy(), 1.0, lam.copy()
    for _ in range(iters):
        new = np.maximum(y - (G @ y - h) / max(L, 1e-12), 0.0)
        t_new = (1 + math.sqrt(1 + 4 * t * t)) / 2
        y = new + (t - 1) / t_new * (new - lam)
        if np.abs(new - lam).max() < 1e-14:
            lam = new
            break
        lam, t = new, t_new
    return lam


def solve_qp(v, A, c, w=None, n_outer=8):
    """Closest u to v with A u + w r >= c (slack r penalized by ||r||^2). Working-set method:
    solve the dual on the currently violated rows, add rows the solution violates, repeat."""
    keep = (A ** 2).sum(1) > 1e-14
    A, c = A[keep], c[keep]
    w2 = np.zeros(len(c)) if w is None else (w[keep] ** 2)
    viol = c - A @ v > _TOL
    if not viol.any():
        return v
    W = viol.copy()
    u = v
    for _ in range(n_outer):
        idx = np.where(W)[0]
        Aw = A[idx]
        G, h = Aw @ Aw.T + np.diag(w2[idx]), c[idx] - Aw @ v
        lam = _dual_active_set(G, h)
        rho = h - G @ lam
        if lam.min() < -_TOL or rho.max() > _TOL:  # active set failed to certify: projected-gradient fallback
            lam = _fista(G, h, np.maximum(lam, 0.0))
        u = v + Aw.T @ lam
        lam_full = np.zeros(len(c))
        lam_full[idx] = lam
        new = (c - A @ u - w2 * lam_full > _TOL) & ~W
        if not new.any():
            break
        W |= new
    return u


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------

def _filter(constraint, x, v, t, cond, lam, eps, rho, delta, state, slack, t_w, w_scale):
    x_eval = x + (1.0 - t) * v if state == "tweedie" else x
    with torch.enable_grad():
        b, A = constraint.barrier_rows(x_eval)
    if state == "tweedie":
        A = A * (1.0 - t)
    if cond == "expK":
        c = -lam * (b - delta)
    else:
        z = b - delta
        c = -eps * torch.sign(z) * z.abs().pow(rho)
    if slack == "sfm":
        w_t = w_scale * (1.0 - math.exp(3.0 * (t / t_w - 1.0))) if t < t_w else 0.0
    else:
        w_t = float(slack)
    B = x.shape[0]
    vf = v.reshape(B, -1).detach().cpu().double().numpy()
    An, cn = A.detach().cpu().double().numpy(), c.detach().cpu().double().numpy()
    u = np.stack([solve_qp(vf[i], An[i], cn[i], np.full(cn.shape[1], w_t) if w_t > 0 else None) for i in range(B)])
    return torch.as_tensor(u, dtype=v.dtype, device=v.device).view_as(v)


@torch.no_grad()
def safediffuser(policy, constraint, n_steps=10, lam=1.0, delta=0.0, state="ode", slack=0.0):
    """SafeDiffuser (Xiao et al., 2023), exponential class-K barrier condition at every Euler step."""
    nfe0, x, dt, t = policy.nfe, policy.noise(), 1.0 / n_steps, 0.0
    for _ in range(n_steps):
        v = policy.velocity(x, t)
        x = x + _filter(constraint, x, v, t, "expK", lam, 0.0, 1.0, delta, state, slack, 0.9, 200.0) * dt
        t += dt
    return x, dict(nfe=policy.nfe - nfe0)


@torch.no_grad()
def safeflowmatcher(policy, constraint, T_p=1, T_c=10, alpha=2.0, eps=0.5, rho=0.9, delta=0.0, state="ode", slack=0.0,
                    t_w=0.9, w_scale=200.0):
    """SafeFlowMatcher (Algorithm 1): prediction pass, then a filtered correction pass with the
    vanishing time-scaled field alpha (1 - t) v and the finite-time barrier condition."""
    nfe0, x = policy.nfe, policy.noise()
    dt, t = 1.0 / T_p, 0.0
    for _ in range(T_p):
        x = x + policy.velocity(x, t) * dt
        t += dt
    dt, t = 1.0 / T_c, 0.0
    for _ in range(T_c):
        v = alpha * (1.0 - t) * policy.velocity(x, t)
        x = x + _filter(constraint, x, v, t, "finite_time", 0.0, eps, rho, delta, state, slack, t_w, w_scale) * dt
        t += dt
    return x, dict(nfe=policy.nfe - nfe0)

