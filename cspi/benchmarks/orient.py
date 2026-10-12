"""3D-Orient: reach the goal while holding the end-effector orientation at a fixed top-down grasp.

Policy     : FlowUNet on 16 x 7 absolute joint chunks, obs = (q_current, q_goal), receding horizon
             (8 executed steps per plan, <= 16 plans, stop within 0.1 rad of the goal).
Constraint : orientation cost  lambda_o sum_t ||R_EE(q_t) - R*||_F^2  (a soft lock; it never reaches 0).
CS-Pi's P  : proximal L-BFGS step on the orientation cost plus an end-effector proximity term
             lambda_ee sum_t ||EE(q_t) - EE_ref_t||^2 that keeps the hand on the prior's own path.
             EE_ref is the end-effector path of one unguided rollout from q_start (computed per task).
Guidance / Trust use the orientation cost plus a joint-limit hinge.
Metrics    : success = some step within 10 cm of the goal position with ||R - R*||_F < 0.5;
             goal reaching = some step within 10 cm; max end-effector step (m); NFE; logL.
"""
import time

import torch

from cspi.benchmarks.mpd import (chunk_barrier_rows, jl_barrier, joint_limit_violation, planning_task,
                                 prox_lbfgs, rollout_chunks)
from cspi.flow import ode_solve
from cspi.samplers import Problem

R_TARGET = torch.tensor([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
N_EXEC, MAX_PLANS, GOAL_TOL = 8, 16, 0.1


def ee_pos(robot, q):
    return robot.get_EE_position(q.reshape(-1, q.shape[-1])).reshape(*q.shape[:-1], 3)


def orient_err(robot, q):
    """||R_EE(q_t) - R*||_F^2 per step -> (..., T)."""
    R = robot.get_EE_orientation(q.reshape(-1, q.shape[-1]), rotation_matrix=True).reshape(*q.shape[:-1], 3, 3)
    return ((R - R_TARGET) ** 2).sum(dim=(-1, -2))


class OrientProblem(Problem):

    def __init__(self, model, norm, robot, q_current, q_goal, ee_ref=None, lam_orient=10.0, lam_ee=3000.0,
                 lam_jl=100.0, beta=1e6, proj_steps=5, cbf_groups="orient+jl"):
        obs = torch.cat([q_current, q_goal.view(1, -1).expand(q_current.shape[0], -1)], -1)
        super().__init__(model, norm.normalize_obs(obs), (q_current.shape[0], model.horizon, model.action_dim))
        self.norm, self.robot, self.ee_ref = norm, robot, ee_ref
        self.lam_orient, self.lam_ee, self.lam_jl = lam_orient, lam_ee, lam_jl
        self.beta, self.proj_steps, self.cbf_groups = beta, proj_steps, cbf_groups.split("+")

    def joints(self, x):
        return self.norm.unnormalize(x).cpu()

    def constraint_cost(self, q):
        c = self.lam_orient * orient_err(self.robot, q).sum(-1)
        if self.ee_ref is not None:
            c = c + self.lam_ee * ((ee_pos(self.robot, q) - self.ee_ref) ** 2).sum(dim=(-1, -2))
        return c

    def violation(self, x):
        return self.constraint_cost(self.joints(x))

    def project(self, x, final=False):
        return prox_lbfgs(x, self.joints, self.constraint_cost, self.robot, self.beta,
                          self.proj_steps * (5 if final else 1))

    def cost(self, x):
        q = self.joints(x)
        return self.lam_orient * orient_err(self.robot, q).sum(-1) + self.lam_jl * joint_limit_violation(q, self.robot)

    def barrier_rows(self, x):
        """orient: b = 0.5 - ||R_EE - R*||_F per step (the success band); jl: joint limits."""
        fns = dict(orient=lambda q: 0.5 - orient_err(self.robot, q).clamp(min=1e-18).sqrt().unsqueeze(-1),
                   jl=lambda q: jl_barrier(q, self.robot))
        return chunk_barrier_rows(x, self.norm, lambda q: torch.cat([fns[k](q) for k in self.cbf_groups], -1))


@torch.no_grad()
def reference_path(model, norm, robot, q_start, q_goal, step_size=0.1):
    """End-effector positions of one unguided rollout, one (H, 3) block per plan (EE_ref)."""
    q, refs = q_start.view(1, -1), []
    for _ in range(MAX_PLANS):
        obs_n = norm.normalize_obs(torch.cat([q, q_goal.view(1, -1)], -1))
        v = lambda x, t: model(x, torch.full((1,), t), obs_n)  # noqa: E731
        chunk = norm.unnormalize(ode_solve(v, torch.randn(1, model.horizon, model.action_dim), 0.0, 1.0,
                                           round(1.0 / step_size))).cpu()
        refs.append(ee_pos(robot, chunk)[0])
        q = chunk[:, N_EXEC - 1]
        if float((q[0] - q_goal).norm()) < GOAL_TOL:
            break
    return refs + [refs[-1]] * (MAX_PLANS - len(refs))


@torch.no_grad()
def evaluate(sampler, params, model, norm, tasks, ee_prox=False, seed=0, **problem_kw):
    """One rollout per task (B = 1). Returns per-rollout records incl. chunks for logL."""
    robot = planning_task("spheres").robot
    torch.manual_seed(seed)
    out = []
    for q_s, q_g in zip(tasks["q_start"], tasks["q_goal"]):
        t0 = time.perf_counter()
        refs = reference_path(model, norm, robot, q_s, q_g) if ee_prox else None
        plan = [0]

        def make_problem(q):
            ref = refs[plan[0]] if refs is not None else None
            plan[0] += 1
            return OrientProblem(model, norm, robot, q, q_g, ee_ref=ref, **problem_kw)

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


def evaluate_unit(unit, sampler, params, model, norm, assets, ee_prox=False, n_tasks=None, **problem_kw):
    tasks = torch.load(f"{assets}/tasks/orient.pt")
    sel = tasks["seed"] == int(unit)
    tasks = dict(q_start=tasks["q_start"][sel][:n_tasks], q_goal=tasks["q_goal"][sel][:n_tasks])
    return evaluate(sampler, params, model, norm, tasks,
                    ee_prox=ee_prox, seed=int(unit), **problem_kw)


def summarize(records):
    mean = lambda k: float(sum(float(r[k]) for r in records) / len(records))  # noqa: E731
    return dict(success=mean("success"), goal=mean("goal"), max_ee_step=mean("max_ee_step"), nfe=mean("nfe"),
                joint_limits_ok=mean("joint_limits_ok"), n=len(records))
