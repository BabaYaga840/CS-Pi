"""Sphere-Avoid: plan a whole Panda joint trajectory between two configurations around spheres.

Policy     : FlowUNet on 28 x 7 B-spline control points, obs = (q_start, q_goal), min/max normalized.
             Decoded to a dense 64 x 7 joint trajectory by MPD's degree-5 B-spline.
Env        : EnvSpheres3D (10 spheres, in-distribution) or EnvSpheres3DExtraObjectsV00 (+6, OOD).
Constraint : chosen in the YAML (spline_collision in the paper: MPD collision cost of the dense
             trajectory < tau, projected by Newton steps on the control points).
Context    : cls(policy, task=MPD PlanningTask, cost=MPDCost, q_start=(7,), q_goal=(7,)).
Metrics    : Safe-SR (cost < tau and endpoint error < 0.1 rad), averaged over tasks whose start and
             goal are themselves collision-free; NFE; logL of the control points.
"""
import time

import torch

from cspi.benchmarks.mpd import planning_task
from cspi.constraints.panda import MPDCost
from cspi.constraints.spline_collision import spline_dense

ENVS = {"id": "EnvSpheres3D", "ood": "EnvSpheres3DExtraObjectsV00"}


def trajectory(task, norm, x, q_start, q_goal):
    """Dense trajectory with the start and goal configurations enforced (what is evaluated)."""
    q = spline_dense(task, norm, x, q_start, q_goal).detach().clone()
    q[:, 0], q[:, -1] = q_start, q_goal
    return q


def load(split="id"):
    task = planning_task("spheres", env_id=ENVS[split], num_T_pts=64)
    return task, MPDCost(task, w_obj=10.0, w_self=1.0, w_jl=1.0)


@torch.no_grad()
def evaluate(sampler, params, policy, make_constraint, tasks, split="id", batch=8, tau=1e-3, seed=0):
    """Run one method on every task; returns per-task records (samples kept for logL scoring)."""
    task, cost = load(split)
    torch.manual_seed(seed)
    out = []
    for q_s, q_g in zip(tasks["q_start"], tasks["q_goal"]):
        t0 = time.perf_counter()
        feasible = float(cost(q_s.view(1, 1, -1))) < tau and float(cost(q_g.view(1, 1, -1))) < tau
        p = policy.condition(torch.cat([q_s, q_g]).view(1, -1).expand(batch, -1))
        x, info = sampler(p, make_constraint(p, task=task, cost=cost, q_start=q_s, q_goal=q_g), **params)
        q = trajectory(task, p.normalizer, x, q_s, q_g)
        c = cost(q)
        goal_err = (q[:, -1] - q_g).norm(dim=-1)
        out.append(dict(endpoints_feasible=feasible, safe=((c < tau) & (goal_err < 0.1)).float().mean().item(),
                        collision_free=(c < tau).float().mean().item(), nfe=info["nfe"],
                        n_history=info.get("n_history", []), fallback=bool(info.get("fallback", False)),
                        x=x.cpu(), obs=p.obs_n.cpu(), wall_s=time.perf_counter() - t0))
    return out


def evaluate_unit(unit, sampler, params, policy, make_constraint, assets, batch=8, tau=1e-3, seed=0, n_tasks=None):
    """unit: "id" (10 spheres) or "ood" (16 spheres)."""
    tasks = {k: v[:n_tasks] for k, v in torch.load(f"{assets}/tasks/spheres.pt").items() if torch.is_tensor(v)}
    records = evaluate(sampler, params, policy, make_constraint, tasks, unit, batch, tau, seed)
    for r in records:
        r["score"] = r["endpoints_feasible"]  # logL over tasks with collision-free endpoints
    return records


def summarize(records):
    feas = [r for r in records if r["endpoints_feasible"]]
    mean = lambda rs, k: float(sum(float(r[k]) for r in rs) / max(len(rs), 1))  # noqa: E731
    return dict(success=mean(feas, "safe"), collision_free=mean(feas, "collision_free"),
                nfe=mean(records, "nfe"), fallback_rate=mean(records, "fallback"),
                n_tasks=len(records), n_tasks_feasible=len(feas))
