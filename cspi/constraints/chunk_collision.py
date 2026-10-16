"""Collision-free joint chunks in a scene with obstacle boxes (Shelf pick-and-place).

x is (B, 16, 7) normalized absolute joint chunks.

violation    : MPD collision cost (scene, scenario boxes, self, joint limits) of the chunk.
P            : proximal L-BFGS step  argmin c(x') + JL(x') + ||x' - x||^2 / (2 beta).
post_project : surface sliding (`slide`), used after guidance / Trust with `post_project: true`.

Context supplied by the benchmark: task (MPD PlanningTask), cost (MPDCost incl. the scenario boxes),
q_current (B, 7), boxes (analytic boxes from `box_groups`, for the CBF barriers).
"""
import torch

from cspi.constraints.base import Constraint
from cspi.constraints.panda import box_barrier, chunk_barrier_rows, jl_barrier, prox_lbfgs, self_barrier, slide


class ChunkCollision(Constraint):
    """proj_steps: L-BFGS iterations per projection (x5 for the terminal one and for slide);
    beta: proximal weight; tau: slide stops once cost and joint-limit violation are below it;
    cbf_groups: barrier groups for the CBF baselines ("obs", "jl", "self")."""

    def __init__(self, policy, task, cost, q_current, boxes=None, proj_steps=6, beta=1.0, tau=1e-3,
                 cbf_groups="obs+jl"):
        super().__init__(policy)
        self.task, self.robot, self.cost_fn, self.q_current = task, task.robot, cost, q_current
        self.proj_steps, self.beta, self.tau, self.boxes = proj_steps, beta, tau, boxes
        self.cbf_groups = cbf_groups.split("+")

    def joints(self, x):
        return self.norm.unnormalize(x).cpu()

    def violation(self, x):
        return self.cost_fn(self.joints(x))

    def project(self, x, final=False):
        return prox_lbfgs(x, self.joints, self.cost_fn, self.robot, self.beta, self.proj_steps * (5 if final else 1))

    def cost(self, x):
        return self.cost_fn(self.joints(x))

    def barrier_rows(self, x):
        fns = dict(obs=lambda q: box_barrier(q, self.task, self.boxes), jl=lambda q: jl_barrier(q, self.robot),
                   self=lambda q: self_barrier(q, self.task))
        return chunk_barrier_rows(x, self.norm, lambda q: torch.cat([fns[k](q) for k in self.cbf_groups], -1))

    def post_project(self, x):
        """Surface sliding, with the chunk's first waypoint pinned to the current joint state."""
        lo, hi = self.norm.x_a, self.norm.x_b
        anchor = (((self.q_current - lo) / (hi - lo).clamp(min=1e-6)) * 2.0 - 1.0).clamp(-1.0, 1.0)
        return slide(x, self.joints, self.cost_fn, self.robot, anchor, self.tau, n_iter=self.proj_steps * 5)
