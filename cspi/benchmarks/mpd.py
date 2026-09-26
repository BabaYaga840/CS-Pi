"""Shared glue for the three Franka Panda benchmarks, built on MPD (Carvalho et al.).

MPD supplies the robot (forward kinematics, collision spheres, joint limits), the
environments and their signed-distance collision fields. Install it from
https://github.com/joaoamcarvalho/mpd-splines-public at commit 3676cbf, apply
third_party/mpd.patch, and point MPD_ROOT at the checkout. Everything here runs on
CPU (the FK library keeps internal CPU tensors).
"""
import os
import sys

import numpy as np
import torch

MPD_ROOT = os.environ.get("MPD_ROOT", os.path.expanduser("~/repos/mpd-splines-public"))
for _p in (MPD_ROOT, os.path.join(MPD_ROOT, "mpd", "torch_robotics")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mpd.utils.patches import numpy_monkey_patch  # noqa: E402
import torchkin.forward_kinematics as _tk_fk  # noqa: E402

numpy_monkey_patch()


def _no_leak_factory(*args, _factory=_tk_fk.ForwardKinematicsFactory, **kwargs):
    """torchkin keeps its forward outputs on the autograd context (`ctx.poses = output[-1]`), a reference
    cycle that leaks every differentiable FK call. Backward only reads their values, so keep detached copies."""
    out = _factory(*args, **kwargs)

    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(*output[:-1])
        ctx.poses = [p.detach() if p is not None else None for p in output[-1]]

    out[0].setup_context = staticmethod(setup_context)
    return out


_tk_fk.ForwardKinematicsFactory = _no_leak_factory
from mpd.parametric_trajectory.trajectory_bspline import ParametricTrajectoryBspline  # noqa: E402
from torch_robotics import environments, robots  # noqa: E402
from torch_robotics.tasks.tasks import PlanningTask  # noqa: E402

CPU = dict(device=torch.device("cpu"), dtype=torch.float32)

# Collision margins of the MPD datasets the priors were trained on (their args.yaml).
DATASET_ARGS = {
    "spheres": dict(env_id="EnvSpheres3D", robot_id="RobotPanda", min_distance_robot_env=0.02),
    "shelf": dict(env_id="EnvTableShelf", robot_id="RobotPanda", min_distance_robot_env=0.0),
}


def planning_task(dataset, env_id=None, num_T_pts=64, n_control_points=28):
    """Build the MPD PlanningTask exactly as mpd.utils.loaders does, without loading the dataset.

    n_control_points=28 is MPD's adjustment of 24 requested points (degree-5 B-spline,
    zero velocity and acceleration at both ends).
    """
    a = DATASET_ARGS[dataset]
    env = getattr(environments, env_id or a["env_id"])(tensor_args=CPU)
    robot = getattr(robots, a["robot_id"])(tensor_args=CPU)
    traj = ParametricTrajectoryBspline(
        n_control_points=n_control_points, degree=5, zero_vel_at_start_and_goal=True,
        zero_acc_at_start_and_goal=True, num_T_pts=num_T_pts, remove_outer_control_points=False,
        keep_last_control_point=False, trajectory_duration=5.0, phase_time_class="PhaseTimeLinear",
        phase_time_args={}, tensor_args=CPU)
    return PlanningTask(env=env, robot=robot, parametric_trajectory=traj, tensor_args=CPU,
                        min_distance_robot_env=a["min_distance_robot_env"],
                        obstacle_cutoff_margin=a["min_distance_robot_env"],
                        margin_for_dense_collision_checking=0.0)


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
# Chunk projections (normalized absolute joint chunks, CPU)
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
# Receding-horizon rollout of a chunk policy (Orient, Shelf)
# ---------------------------------------------------------------------------

@torch.no_grad()
def rollout_chunks(make_problem, sampler, params, q_start, batch, n_exec, max_plans,
                   q_goal=None, goal_tol=0.0, max_dq=0.0, post=None):
    """Plan a chunk from the current joint state, execute its first n_exec steps, repeat.

    make_problem(q_current (B, 7)) -> Problem.  A sample that gets within goal_tol of q_goal
    freezes in place; the rollout ends when all are frozen or after max_plans plans.
    max_dq > 0 rate-limits executed joints (rad per step) relative to the last executed state.
    post(problem, x) optionally post-processes each sampled chunk.
    """
    q = q_start.view(1, -1).expand(batch, -1).clone()
    executed, frozen = [q], torch.zeros(batch, dtype=torch.bool)
    chunks, obs, infos, nfe = [], [], [], 0
    for plan in range(max_plans):
        if q_goal is not None and plan > 0:
            frozen = frozen | ((q - q_goal.view(1, -1)).norm(dim=-1) < goal_tol)
            if bool(frozen.all()):
                break
        p = make_problem(q)
        x, info = sampler(p, **params)
        if post is not None:
            x = post(p, x)
        nfe += info["nfe"]
        infos.append(info)
        chunks.append(x.detach().cpu())
        obs.append(p.obs_n.detach().cpu())
        chunk = p.joints(x)
        for i in range(n_exec):
            target = chunk[:, i]
            if max_dq > 0:
                target = executed[-1] + (target - executed[-1]).clamp(-max_dq, max_dq)
            executed.append(torch.where(frozen.view(-1, 1), q, target))
        q = torch.where(frozen.view(-1, 1), q, executed[-1])
    return dict(q_traj=torch.stack(executed[1:], dim=1), chunks=torch.stack(chunks), obs=torch.stack(obs),
                infos=infos, nfe=nfe, n_plans=len(infos))


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
