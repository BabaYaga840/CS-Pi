"""3D-Orient: reach the goal while holding the end-effector orientation at a fixed top-down grasp.

Policy     : FlowUNet on 16 x 7 absolute joint chunks, obs = (q_current, q_goal), receding horizon
             (8 executed steps per plan, <= 16 plans, stop within 0.1 rad of the goal).
Constraint : chosen in the YAML (orientation_lock in the paper).
Context    : cls(policy, robot=RobotPanda, q_current=(B, 7), ee_ref=(H, 3) or None). With
             eval `ee_prox: true`, ee_ref is the end-effector path of one unguided rollout from
             q_start (computed per task, one (H, 3) block per plan), which turns on the EE-prox term.
Metrics    : success = some step within 10 cm of the goal position with ||R - R*||_F < 0.5;
             goal reaching = some step within 10 cm; max end-effector step (m); NFE; logL.
"""
import time

import torch

from cspi.benchmarks.mpd import planning_task, rollout_chunks
from cspi.constraints.orientation import ee_pos, orient_err
from cspi.constraints.panda import joint_limit_violation

N_EXEC, MAX_PLANS, GOAL_TOL = 8, 16, 0.1


def _obs(q, q_goal):
    return torch.cat([q, q_goal.view(1, -1).expand(q.shape[0], -1)], -1)


@torch.no_grad()
def reference_path(policy, robot, q_start, q_goal, step_size=0.1):
    """End-effector positions of one unguided rollout, one (H, 3) block per plan (EE_ref)."""
    q, refs = q_start.view(1, -1), []
    for _ in range(MAX_PLANS):
        p = policy.condition(_obs(q, q_goal))
        chunk = p.normalizer.unnormalize(p.sample(step_size)).cpu()
        refs.append(ee_pos(robot, chunk)[0])
        q = chunk[:, N_EXEC - 1]
        if float((q[0] - q_goal).norm()) < GOAL_TOL:
            break
    return refs + [refs[-1]] * (MAX_PLANS - len(refs))


@torch.no_grad()
def evaluate(sampler, params, policy, make_constraint, tasks, ee_prox=False, seed=0):
    """One rollout per task (B = 1). Returns per-rollout records incl. chunks for logL."""
    robot = planning_task("spheres").robot
    torch.manual_seed(seed)
    out = []
    for q_s, q_g in zip(tasks["q_start"], tasks["q_goal"]):
        t0 = time.perf_counter()
        refs = reference_path(policy, robot, q_s, q_g) if ee_prox else None
        plan = [0]

        def make_problem(q):
            ref = refs[plan[0]] if refs is not None else None
            plan[0] += 1
            p = policy.condition(_obs(q, q_g))
            return p, make_constraint(p, robot=robot, q_current=q, ee_ref=ref)

        r = rollout_chunks(make_problem, sampler, params, q_s, 1, N_EXEC, MAX_PLANS, q_goal=q_g, goal_tol=GOAL_TOL)
        q = torch.cat([q_s.view(1, 1, -1), r["q_traj"]], dim=1)[0]
        ee = ee_pos(robot, q)
        dist = (ee - ee_pos(robot, q_g.view(1, -1))[0]).norm(dim=-1)
        frob = orient_err(robot, q).sqrt()
        out.append(dict(success=bool(((dist < 0.10) & (frob < 0.5)).any()), goal=bool((dist < 0.10).any()),
                        max_ee_step=float((ee[1:] - ee[:-1]).norm(dim=-1).max()), nfe=r["nfe"],
                        joint_limits_ok=float(joint_limit_violation(q[None], robot)) < 1e-3,
                        n_history=[i.get("n_history", []) for i in r["infos"]],
                        fallback_flags=[bool(i.get("fallback", False)) for i in r["infos"]], q_traj=q,
                        x=r["chunks"][:, 0], obs=r["obs"][:, 0], wall_s=time.perf_counter() - t0))
    return out


def evaluate_unit(unit, sampler, params, policy, make_constraint, assets, ee_prox=False, n_tasks=None):
    """unit: the task seed (30 tasks per seed); the torch seed is the same."""
    tasks = torch.load(f"{assets}/tasks/orient.pt")
    sel = tasks["seed"] == int(unit)
    tasks = dict(q_start=tasks["q_start"][sel][:n_tasks], q_goal=tasks["q_goal"][sel][:n_tasks])
    return evaluate(sampler, params, policy, make_constraint, tasks, ee_prox=ee_prox, seed=int(unit))


def summarize(records):
    mean = lambda k: float(sum(float(r[k]) for r in records) / len(records))  # noqa: E731
    return dict(success=mean("success"), goal=mean("goal"), max_ee_step=mean("max_ee_step"), nfe=mean("nfe"),
                joint_limits_ok=mean("joint_limits_ok"), n=len(records))
