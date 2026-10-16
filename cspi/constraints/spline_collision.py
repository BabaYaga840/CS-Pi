"""Collision-free whole-trajectory plans represented by B-spline control points (Sphere-Avoid).

x is (B, 28, 7) normalized control points; MPD's degree-5 B-spline decodes them to a dense
(B, 64, 7) joint trajectory between q_start and q_goal.

violation : MPD collision cost (obstacles, self-collision, joint limits) of the dense trajectory.
P         : Newton steps on the control points, x <- x - c(x) grad / ||grad||^2, with the dense-space
            gradient mapped to control points through the B-spline basis; stops once c <= tau.

Context supplied by the benchmark: task (MPD PlanningTask), cost (MPDCost), q_start, q_goal (7,).
"""
import torch

from cspi.constraints.base import Constraint
from cspi.constraints.panda import jl_barrier, self_barrier, sphere_barrier, waypoint_jacobian

N_PINNED = 3  # control points fixed by the boundary conditions at each end


def spline_dense(task, norm, x, q_start, q_goal):
    """Dense joint trajectory (B, T, 7) of normalized control points x (B, H, 7)."""
    ctrl = norm.unnormalize(x.cpu())
    return task.parametric_trajectory.get_q_trajectory(ctrl, q_start, q_goal, get_type=("pos",))["pos"]


class SplineCollision(Constraint):
    """proj_steps: Newton steps per projection (x5 for the terminal one); tau: projection stops once
    the cost is below it; cbf_groups: barrier groups for the CBF baselines ("obs", "jl", "self")."""

    def __init__(self, policy, task, cost, q_start, q_goal, proj_steps=10, tau=1e-3, cbf_groups="obs+jl"):
        super().__init__(policy)
        self.task, self.cost_fn = task, cost
        self.q_start, self.q_goal, self.proj_steps, self.tau = q_start, q_goal, proj_steps, tau
        self.cbf_groups = cbf_groups.split("+")
        self.grad_mask = torch.ones(1, *policy.shape[1:])
        self.grad_mask[:, :N_PINNED] = 0.0
        self.grad_mask[:, -N_PINNED:] = 0.0
        with torch.no_grad():  # the decoder is affine in x; its basis maps dense gradients to control points
            H, d = policy.shape[1:]
            base = self.dense(torch.zeros(1, H, d))[0]
            self.basis = torch.stack([self.dense(torch.eye(H)[i].view(1, H, 1).expand(1, H, d))[0] - base
                                      for i in range(H)])[:, :, 0]  # (H, T)

    def dense(self, x):
        return spline_dense(self.task, self.norm, x, self.q_start, self.q_goal)

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
            H, d = self.policy.shape[1:]
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
