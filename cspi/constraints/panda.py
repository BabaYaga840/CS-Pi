"""Constraint tools shared by the three Franka Panda benchmarks (built on MPD's robot and collision fields).

Nothing here imports MPD: the robot, planning task and collision fields are passed in by the
benchmark (cspi/benchmarks/mpd.py builds them). Everything runs on CPU (the FK library keeps
internal CPU tensors).

MPDCost              differentiable collision + joint-limit cost of a joint trajectory, 0 iff feasible
prox_lbfgs           proximal projection  argmin_x' cost(q(x')) + JL + ||x' - x||^2 / (2 beta)
slide                surface-sliding projection (Shelf post-projection for guidance / Trust)
*_barrier, chunk_barrier_rows   barrier functions for the CBF baselines
"""
import numpy as np
import torch

CPU = dict(device=torch.device("cpu"), dtype=torch.float32)


def joint_limit_violation(q, robot):
    """Hinge sum_{t,j} [q_min - q]_+ + [q - q_max]_+  -> (B,)."""
    return (torch.relu(robot.q_pos_min - q) + torch.relu(q - robot.q_pos_max)).sum(dim=(-1, -2))


def _with_limits(cost, q, robot):
    """cost(q) plus the lower and upper joint-limit hinges (summed separately, as in the paper runs)."""
    return (cost(q) + torch.relu(robot.q_pos_min - q).sum(dim=(1, 2))
            + torch.relu(q - robot.q_pos_max).sum(dim=(1, 2)))


class MPDCost:
    """Differentiable constraint cost of a joint trajectory q (B, T, 7) -> (B,), zero iff feasible:

        w_obj * (environment obstacles + extra obstacles) + w_self * self-collision
        + workspace boundary + w_jl * joint-limit hinge  (+ w_obj * scenario obstacles)
    """

    def __init__(self, task, w_obj=10.0, w_self=1.0, w_jl=1.0, extra_field=None):
        self.task, self.robot = task, task.robot
        self.w_obj, self.w_self, self.w_jl, self.extra_field = w_obj, w_self, w_jl, extra_field

    def __call__(self, q):
        t = self.task
        fk = self.robot.fk_map_collision(q)
        c = torch.zeros(q.shape[0], **CPU)
        if t.df_collision_objects is not None:
            c = c + self.w_obj * t.df_collision_objects.compute_cost(q, fk, field_type="sdf").sum(-1)
        if t.df_collision_self is not None:
            c = c + self.w_self * t.df_collision_self.compute_cost(q, fk, field_type="sdf").sum(-1)
        if getattr(t, "df_collision_extra_objects", None) is not None and getattr(t.env, "obj_extra_list", None):
            c = c + self.w_obj * t.df_collision_extra_objects.compute_cost(q, fk, field_type="sdf").sum(-1)
        if t.df_collision_ws_boundaries is not None:
            c = c + t.df_collision_ws_boundaries.compute_cost(q, fk, field_type="sdf").sum(-1)
        c = c + self.w_jl * joint_limit_violation(q, self.robot)
        if self.extra_field is not None:
            fk = self.robot.fk_map_collision(q)
            c = c + self.w_obj * self.extra_field.compute_cost(q, fk, field_type="sdf").sum(-1)
        return c


# ---------------------------------------------------------------------------
# Projections (normalized joint chunks, CPU)
# ---------------------------------------------------------------------------

def prox_lbfgs(x, joints, cost, robot, beta=1.0, n_iter=10):
    """P(x) = argmin_x' cost(q(x')) + JL(q(x')) + ||x' - x||^2 / (2 beta)  (L-BFGS, strong Wolfe)."""
    x0 = x.detach().cpu().clone()

    def objective(z):
        return (_with_limits(cost, joints(z), robot) + ((z - x0) ** 2).sum(dim=(1, 2)) / (2.0 * beta)).sum()

    z = x0.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([z], lr=1.0, max_iter=n_iter, line_search_fn="strong_wolfe", history_size=10)

    def closure():
        opt.zero_grad()
        f = objective(z)
        f.backward()
        return f

    with torch.enable_grad():
        try:
            opt.step(closure)
        except RuntimeError:  # rare line-search failure: fall back to normalized gradient descent
            z = x0.clone()
            for _ in range(n_iter):
                zz = z.detach().requires_grad_(True)
                g, = torch.autograd.grad(objective(zz), zz)
                z = (zz - 0.1 * g / (g.norm() + 1e-8)).detach().clamp(-1.5, 1.5)
            return z.to(x.device)
    return z.detach().clamp(-1.5, 1.5).to(x.device)


def slide(x, joints, cost, robot, q_anchor_n, tau=1e-3, n_iter=50, step=0.06, w_tangent=1.0, w_smooth=0.15):
    """Surface-sliding projection used after guidance / Trust on Shelf.

    Waypoints in contact move out along the cost normal plus the component of the pull back to
    the original chunk that is tangent to the obstacle surface; a Laplacian smooths the chunk,
    its first waypoint is pinned to the current joint state, and the lowest-cost iterate is returned.
    """
    z = x.detach().cpu().clone()
    z0, best = z.clone(), z.clone()
    best_c = torch.full((z.shape[0],), float("inf"))
    for _ in range(n_iter):
        with torch.enable_grad():
            zz = z.detach().requires_grad_(True)
            q = joints(zz)
            c = cost(q)
            lo = torch.relu(robot.q_pos_min - q).sum(dim=(1, 2))
            hi = torch.relu(q - robot.q_pos_max).sum(dim=(1, 2))
            viol = c + lo + hi
            g, = torch.autograd.grad(viol.sum(), zz)
        better = viol.detach() < best_c
        best[better], best_c[better] = z[better], viol.detach()[better]
        if float(c.max()) < tau and float((lo + hi).max()) < tau:
            break
        gn = g.norm(dim=2, keepdim=True)
        n_hat = g / gn.clamp(min=1e-8)
        d = z0 - z
        upd = -n_hat + w_tangent * (d - (d * n_hat).sum(2, keepdim=True) * n_hat)
        z = z + step * (gn > 1e-6).float() * upd
        z[:, 1:-1] = z[:, 1:-1] + w_smooth * (z[:, :-2] - 2.0 * z[:, 1:-1] + z[:, 2:])
        z[:, 0] = q_anchor_n
        z = z.clamp(-1.0, 1.0)
    return best.to(x.device)


# ---------------------------------------------------------------------------
# Barrier functions for the CBF baselines, per waypoint: q (B, T, 7) -> b (B, T, M)
# ---------------------------------------------------------------------------

def jl_barrier(q, robot):
    return torch.cat([q - robot.q_pos_min, robot.q_pos_max - q], dim=-1)


def self_barrier(q, task):
    """min over checked link-sphere pairs of (distance - radii) - cutoff margin."""
    df = task.df_collision_self
    i1, i2 = torch.as_tensor(np.asarray(df.link_idx_1)), torch.as_tensor(np.asarray(df.link_idx_2))
    rr = df.link_radii_1.reshape(-1) + df.link_radii_2.reshape(-1)
    p = task.robot.fk_map_collision(q)
    d = torch.linalg.norm(p[..., i1, :] - p[..., i2, :], dim=-1) - rr - float(df.cutoff_margin)
    return d.min(dim=-1, keepdim=True).values


def sphere_barrier(q, task, centers, radii):
    """One row per obstacle sphere: min over robot spheres of (distance - r_link) - R - cutoff margin
    (b >= 0 exactly where the obstacle term of MPDCost is zero)."""
    df = task.df_collision_objects
    p = task.robot.fk_map_collision(q)
    dist = torch.linalg.norm(p.unsqueeze(3) - centers.view(1, 1, 1, -1, 3), dim=-1)
    dist = dist - torch.as_tensor(df.collision_margins, dtype=torch.float32).reshape(1, 1, -1, 1)
    return (dist - radii.view(1, 1, 1, -1) - float(df.cutoff_margin)).min(dim=2).values


def box_groups(df):
    """Analytic boxes (pose, centers, half sizes, cutoff) behind a collision field, looking through
    MPD's precomputed grid SDF to the original primitives."""
    out, cutoff = [], float(getattr(df, "cutoff_margin", 0.0) or 0.0)
    for o in df.df_obj_list_fn():
        for f in getattr(o, "obj_list", None) or [o]:
            R = f.ori.float() if getattr(f, "ori", None) is not None else torch.eye(3)
            t = f.pos.float() if getattr(f, "pos", None) is not None else torch.zeros(3)
            for prim in f.fields:
                out.append((R, t, prim.centers.float().reshape(-1, 3), prim.half_sizes.float().reshape(-1, 3), cutoff))
    return out


def box_barrier(q, task, groups):
    """One row per box: min over robot spheres of (Chebyshev box SDF - r_link) - cutoff."""
    p = task.robot.fk_map_collision(q)
    r_link = torch.as_tensor(task.df_collision_objects.collision_margins, dtype=torch.float32).reshape(1, 1, -1, 1)
    vals = []
    for R, t, c, h, cutoff in groups:
        d = (torch.matmul(p - t, R).unsqueeze(-2) - c).abs() - h
        sdf = d.max(dim=-1).values - r_link
        vals.append(sdf.min(dim=2).values - cutoff)
    return torch.cat(vals, dim=-1)


def waypoint_jacobian(fn, q):
    """b = fn(q) (B, T, M) with row (t, m) depending on q_t only -> (b, db/dq_t (B, T, M, 7))."""
    with torch.enable_grad():
        qq = q.detach().clone().requires_grad_(True)
        b = fn(qq)
        g = torch.stack([torch.autograd.grad(b[..., m].sum(), qq, retain_graph=True)[0]
                         for m in range(b.shape[-1])], dim=2)
    return b.detach(), g


def chunk_barrier_rows(x, norm, fn, grad_tol=1e-6):
    """Barrier rows of an absolute joint chunk q_k = x_k * std + mean (row (k, m) depends on x_k only).
    Rows whose joint-space gradient is numerically zero cannot be steered and are dropped."""
    half = (norm.x_b - norm.x_a).clamp(min=1e-6) / 2.0
    b, g = waypoint_jacobian(fn, (x * half + (norm.x_a + half)).cpu())
    g = g * (g.norm(dim=-1, keepdim=True) > grad_tol)
    B, H, M = b.shape
    A = torch.eye(H)[None, :, None, :, None] * (g * half.cpu()).unsqueeze(3)
    return b.reshape(B, H * M), A.reshape(B, H * M, -1)
