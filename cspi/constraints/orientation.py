"""Hold the end-effector orientation at a fixed top-down grasp along a joint chunk (3D-Orient).

x is (B, 16, 7) normalized absolute joint chunks.

violation : lam_orient sum_t ||R_EE(q_t) - R*||_F^2  (a soft lock; it never reaches 0), plus, when the
            benchmark supplies an end-effector reference path ee_ref, lam_ee sum_t ||EE(q_t) - EE_ref_t||^2,
            which keeps the hand on the prior's own path ("EE-prox").
P         : proximal L-BFGS step on that cost, argmin cost + JL + ||x' - x||^2 / (2 beta).
cost      : orientation cost plus lam_jl times the joint-limit hinge (guidance / Trust).

Context supplied by the benchmark: robot (MPD RobotPanda), q_current (B, 7), ee_ref (H, 3) or None.
"""
import torch

from cspi.constraints.base import Constraint
from cspi.constraints.panda import chunk_barrier_rows, jl_barrier, joint_limit_violation, prox_lbfgs

R_TARGET = torch.tensor([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])


def ee_pos(robot, q):
    return robot.get_EE_position(q.reshape(-1, q.shape[-1])).reshape(*q.shape[:-1], 3)


def orient_err(robot, q):
    """||R_EE(q_t) - R*||_F^2 per step -> (..., T)."""
    R = robot.get_EE_orientation(q.reshape(-1, q.shape[-1]), rotation_matrix=True).reshape(*q.shape[:-1], 3, 3)
    return ((R - R_TARGET) ** 2).sum(dim=(-1, -2))


class OrientationLock(Constraint):
    """lam_orient, lam_ee, lam_jl: weights of the orientation, EE-proximity and joint-limit terms;
    beta: proximal weight of P (large = trust the cost, small = stay near x); proj_steps: L-BFGS
    iterations per projection (x5 for the terminal one); cbf_groups: "orient", "jl"."""

    def __init__(self, policy, robot, q_current, ee_ref=None, lam_orient=10.0, lam_ee=3000.0, lam_jl=100.0,
                 beta=1e6, proj_steps=5, cbf_groups="orient+jl"):
        super().__init__(policy)
        self.robot, self.ee_ref = robot, ee_ref
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
