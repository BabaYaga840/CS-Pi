"""Sphere-Avoid: plan a whole Panda joint trajectory between two configurations around spheres.

Policy     : FlowUNet on 28 x 7 B-spline control points, obs = (q_start, q_goal), min/max normalized.
             Decoded to a dense 64 x 7 joint trajectory by MPD's degree-5 B-spline.
Env        : EnvSpheres3D (10 spheres, in-distribution) or EnvSpheres3DExtraObjectsV00 (+6, OOD).
Constraint : MPD collision cost (spheres, self-collision, joint limits) of the dense trajectory < tau.
P          : Newton steps on the control points, x <- x - c(x) grad / ||grad||^2, with the dense-space
             gradient mapped to control points through the B-spline basis.
Metrics    : Safe-SR (cost < tau and endpoint error < 0.1 rad), averaged over tasks whose start and
             goal are themselves collision-free; NFE; logL of the control points.
"""
import time

import torch

from cspi.benchmarks.mpd import MPDCost, jl_barrier, planning_task, self_barrier, sphere_barrier, waypoint_jacobian
from cspi.samplers import Problem

ENVS = {"id": "EnvSpheres3D", "ood": "EnvSpheres3DExtraObjectsV00"}
N_PINNED = 3  # control points fixed by the boundary conditions at each end


class SplineProblem(Problem):

    def __init__(self, model, norm, task, cost, q_start, q_goal, batch=8, proj_steps=10, tau=1e-3,
                 cbf_groups="obs+jl"):
        obs = torch.cat([q_start, q_goal]).view(1, -1).expand(batch, -1)
        super().__init__(model, norm.normalize_obs(obs), (batch, model.horizon, model.action_dim))
        self.norm, self.task, self.cost_fn = norm, task, cost
        self.q_start, self.q_goal, self.proj_steps, self.tau = q_start, q_goal, proj_steps, tau
        self.cbf_groups = cbf_groups.split("+")
        self.grad_mask = torch.ones(1, *self.shape[1:])
        self.grad_mask[:, :N_PINNED] = 0.0
        self.grad_mask[:, -N_PINNED:] = 0.0
        with torch.no_grad():  # the decoder is affine in x; its basis maps dense gradients to control points
            H, d = self.shape[1:]
            base = self.dense(torch.zeros(1, H, d))[0]
            self.basis = torch.stack([self.dense(torch.eye(H)[i].view(1, H, 1).expand(1, H, d))[0] - base
                                      for i in range(H)])[:, :, 0]  # (H, T)

    def dense(self, x):
        ctrl = self.norm.unnormalize(x.cpu())
        return self.task.parametric_trajectory.get_q_trajectory(ctrl, self.q_start, self.q_goal, get_type=("pos",))["pos"]

    def trajectory(self, x):
        """Dense trajectory with the start and goal configurations enforced (what is evaluated)."""
        q = self.dense(x).detach().clone()
        q[:, 0], q[:, -1] = self.q_start, self.q_goal
        return q

    def violation(self, x):
        return self.cost_fn(self.dense(x))

    def project(self, x, final=False):
        z = x.detach().cpu().clone()
        for _ in range(self.proj_steps * (5 if final else 1)):
            with torch.enable_grad():
                q = self.dense(z).detach().requires_grad_(True)
                c = self.cost_fn(q)
                if float(c.max()) <= self.tau:
                    break
                g_dense, = torch.autograd.grad(c.sum(), q)
            g = torch.einsum("ht,btj->bhj", self.basis, g_dense)
            z = (z - c[:, None, None] * g / (g ** 2).sum(dim=(1, 2), keepdim=True).clamp(min=1e-12)).detach()
        return z.to(x.device)

    def cost(self, x):
        return self.cost_fn(self.dense(x))

    def barrier_rows(self, x):
        """Barriers on the 64 dense waypoints (obstacle spheres, joint limits, self-collision), chained
        to the control points through the B-spline basis; pinned control points get no rows."""
        env, robot = self.task.env, self.task.robot
        prims = [f for o in list(env.obj_fixed_list) + list(getattr(env, "obj_extra_list", []) or []) for f in o.fields]
        C = torch.cat([f.centers.float().reshape(-1, 3) for f in prims])
        R = torch.cat([f.radii.float().reshape(-1) for f in prims])
        fns = dict(obs=lambda q: sphere_barrier(q, self.task, C, R),
                   jl=lambda q: jl_barrier(q, robot), self=lambda q: self_barrier(q, self.task))
        b, g = waypoint_jacobian(lambda q: torch.cat([fns[k](q) for k in self.cbf_groups], -1), self.dense(x))
        if not hasattr(self, "basis_ctrl"):  # dq_t / dctrl_j, identical for all joints
            H, d = self.shape[1:]
            f = lambda c: self.task.parametric_trajectory.get_q_trajectory(c, self.q_start, self.q_goal,  # noqa: E731
                                                                           get_type=("pos",))["pos"]
            base = f(torch.zeros(1, H, d))[0, :, 0]
            self.basis_ctrl = torch.stack([f(torch.eye(H)[j].view(1, H, 1).expand(1, H, d))[0, :, 0] - base
                                           for j in range(H)], 1)  # (T, H)
        half = ((self.norm.x_b - self.norm.x_a).clamp(min=1e-6) / 2.0)
        A = torch.einsum("tj,btmd->btmjd", self.basis_ctrl, g) * half
        A[:, :, :, :N_PINNED] = 0.0
        A[:, :, :, -N_PINNED:] = 0.0
        B, T, M = b.shape
        return b.reshape(B, T * M), A.reshape(B, T * M, -1)


def load(split="id"):
    task = planning_task("spheres", env_id=ENVS[split], num_T_pts=64)
    return task, MPDCost(task, w_obj=10.0, w_self=1.0, w_jl=1.0)


@torch.no_grad()
def evaluate(sampler, params, model, norm, tasks, split="id", batch=8, proj_steps=10, tau=1e-3, seed=0,
             cbf_groups="obs+jl"):
    """Run one method on every task; returns per-task records (samples kept for logL scoring)."""
    task, cost = load(split)
    torch.manual_seed(seed)
    out = []
    for q_s, q_g in zip(tasks["q_start"], tasks["q_goal"]):
        t0 = time.perf_counter()
        feasible = float(cost(q_s.view(1, 1, -1))) < tau and float(cost(q_g.view(1, 1, -1))) < tau
        p = SplineProblem(model, norm, task, cost, q_s, q_g, batch, proj_steps, tau, cbf_groups)
        x, info = sampler(p, **params)
        q = p.trajectory(x)
        c = cost(q)
        goal_err = (q[:, -1] - q_g).norm(dim=-1)
        out.append(dict(endpoints_feasible=feasible, safe=((c < tau) & (goal_err < 0.1)).float().mean().item(),
                        collision_free=(c < tau).float().mean().item(), nfe=info["nfe"],
                        n_history=info.get("n_history", []), fallback=bool(info.get("fallback", False)),
                        x=x.cpu(), obs=p.obs_n.cpu(), wall_s=time.perf_counter() - t0))
    return out


def evaluate_unit(unit, sampler, params, model, norm, assets, batch=8, proj_steps=10, tau=1e-3, seed=0,
                  cbf_groups="obs+jl", n_tasks=None):
    tasks = {k: v[:n_tasks] for k, v in torch.load(f"{assets}/tasks/spheres.pt").items() if torch.is_tensor(v)}
    records = evaluate(sampler, params, model, norm, tasks, unit, batch, proj_steps, tau, seed, cbf_groups)
    for r in records:
        r["score"] = r["endpoints_feasible"]  # logL over tasks with collision-free endpoints
    return records


def summarize(records):
    feas = [r for r in records if r["endpoints_feasible"]]
    mean = lambda rs, k: float(sum(float(r[k]) for r in rs) / max(len(rs), 1))  # noqa: E731
    return dict(success=mean(feas, "safe"), collision_free=mean(feas, "collision_free"),
                nfe=mean(records, "nfe"), fallback_rate=mean(records, "fallback"),
                n_tasks=len(records), n_tasks_feasible=len(feas))
