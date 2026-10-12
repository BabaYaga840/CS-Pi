"""Shelf pick-and-place: take a cube out of a shelf cubby and put it down on the table, with new
obstacle boxes on the shelf and table that the demonstrations never saw.

Policy     : FlowUNet on 16 x 7 absolute joint chunks, obs = (q_current, cube position); the place
             location is not conditioned (the prior is multimodal over where to put the cube).
             Receding horizon: 4 samples per task, 8 executed steps per plan, 24 plans, joint rate
             limit 0.3 rad per step.
Constraint : MPD collision cost (shelf, table, self, joint limits) + the scenario boxes < tau.
P          : proximal L-BFGS step  argmin c(x') + JL(x') + ||x' - x||^2 / 2   (6 iterations).
Metrics    : per task, keep the sample with the lowest trajectory cost, then
             Safe-SR  = collision-free and picked and placed,
             goal reaching = picked and placed (kinematic replay of the cube), collision rate; NFE; logL.
"""
import time

import numpy as np
import torch
from torch_robotics.environments.primitives import MultiBoxField, ObjectField
from torch_robotics.torch_planning_objectives.fields.distance_fields import CollisionObjectDistanceField

from cspi.benchmarks.mpd import (CPU, MPDCost, box_barrier, box_groups, chunk_barrier_rows, jl_barrier,
                                 planning_task, prox_lbfgs, rollout_chunks, self_barrier, slide)
from cspi.samplers import Problem

BATCH, N_EXEC, MAX_PLANS, MAX_DQ = 4, 8, 24, 0.30
HAND_TO_CUBE = 0.07
PICK_TOL, PLACE_TOL = 0.08, 0.03
PLACE_REGION = ((0.20, 0.80), (-0.27, 0.27), 0.12)  # table-top box (x, y, z_max) where a place counts


def _on(surface_z, cx, cy, sx, sy, h):
    return (cx, cy, surface_z + h / 2.0), (sx, sy, h)


RUNG = [0.035, 0.265, 0.520]  # shelf board tops
SCENARIOS = {  # cube center, obstacle boxes (center, size)
    "clutter_combo": ((0.50, 0.50, 0.29), [_on(RUNG[1], 0.28, 0.55, 0.10, 0.18, 0.15),
                                           _on(0.0, 0.30, 0.19, 0.12, 0.16, 0.40)]),
    "tall_table_pillar": ((0.30, 0.50, 0.29), [_on(0.0, 0.58, 0.19, 0.12, 0.16, 0.45)]),
    "two_flankers_top": ((0.40, 0.50, 0.545), [_on(RUNG[2], 0.27, 0.55, 0.10, 0.18, 0.12),
                                               _on(RUNG[2], 0.53, 0.55, 0.10, 0.18, 0.12)]),
    "mid_rung_neighbor_E": ((0.40, 0.50, 0.29), [_on(RUNG[1], 0.28, 0.55, 0.10, 0.18, 0.15),
                                                 _on(0.0, 0.66, 0.10, 0.10, 0.14, 0.30)]),
    "low_rung_neighbor_E": ((0.40, 0.50, 0.06), [_on(RUNG[0], 0.52, 0.55, 0.10, 0.18, 0.13),
                                                 _on(0.0, 0.66, 0.10, 0.10, 0.14, 0.30)]),
}


def scenario_cost(task, name):
    """MPDCost of the shelf environment plus the scenario boxes (hard SDF, 1 cm cutoff)."""
    centers = np.array([c for c, _ in SCENARIOS[name][1]], dtype=np.float64)
    sizes = np.array([s for _, s in SCENARIOS[name][1]], dtype=np.float64)
    field = ObjectField([MultiBoxField(centers, sizes, tensor_args=CPU)], "scenario_blocks")
    extra = CollisionObjectDistanceField(task.robot, df_obj_list_fn=lambda: [field],
                                         link_margins_for_object_collision_checking_tensor=task.robot.link_collision_spheres_radii,
                                         cutoff_margin=0.01, tensor_args=CPU)
    return MPDCost(task, w_obj=10.0, w_self=1.0, w_jl=1.0, extra_field=extra)


class ShelfProblem(Problem):

    def __init__(self, model, norm, task, cost, q_current, cube, boxes=None, proj_steps=6, beta=1.0, tau=1e-3,
                 cbf_groups="obs+jl"):
        obs = torch.cat([q_current, cube.view(1, -1).expand(q_current.shape[0], -1)], -1)
        super().__init__(model, norm.normalize_obs(obs), (q_current.shape[0], model.horizon, model.action_dim))
        self.norm, self.task, self.robot, self.cost_fn, self.q_current = norm, task, task.robot, cost, q_current
        self.proj_steps, self.beta, self.tau, self.boxes, self.cbf_groups = proj_steps, beta, tau, boxes, cbf_groups.split("+")

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

    def slide(self, x):
        """Post-projection applied after guidance and Trust Sampling on this benchmark."""
        lo, hi = self.norm.x_a, self.norm.x_b
        anchor = (((self.q_current - lo) / (hi - lo).clamp(min=1e-6)) * 2.0 - 1.0).clamp(-1.0, 1.0)
        return slide(x, self.joints, self.cost_fn, self.robot, anchor, self.tau, n_iter=self.proj_steps * 5)


def pick_and_place(robot, q, cube):
    """Kinematic replay of one joint trajectory q (T, 7): the cube is grasped once the wrist reaches
    the demonstrations' grasp pose (cube - 7 cm along +y) within 8 cm, then carried at the finger
    center until the first step inside the table-top place region. Returns (picked, placed)."""
    pose = robot.get_EE_pose(q)
    pose = pose[:, :3] if pose.shape[-2:] == (4, 4) else pose
    ee, R = pose[:, :3, 3].numpy().astype(np.float64), pose[:, :3, :3].numpy().astype(np.float64)
    grip = ee + R @ np.array([0.0, 0.0, HAND_TO_CUBE])
    cube = np.asarray(cube, dtype=np.float64)
    d_cube = np.linalg.norm(ee - (cube - HAND_TO_CUBE * np.array([0.0, 1.0, 0.0])), axis=-1)
    if not (d_cube < PICK_TOL).any():
        return False, False
    t_pick, T = int(np.argmax(d_cube < PICK_TOL)), q.shape[0]
    carried = np.tile(cube, (T, 1))
    if t_pick + 1 < T:
        (x0, x1), (y0, y1), zmax = PLACE_REGION
        post = grip[t_pick + 1:]
        inside = (post[:, 0] >= x0) & (post[:, 0] <= x1) & (post[:, 1] >= y0) & (post[:, 1] <= y1) & (post[:, 2] <= zmax)
        t_rel = t_pick + 1 + int(np.argmax(inside) if inside.any() else np.argmin(post[:, 2]))
    else:
        t_rel = t_pick
    carried[t_pick:t_rel] = grip[t_pick:t_rel]
    carried[t_rel:] = grip[t_rel - 1] if t_rel > 0 else grip[t_rel]
    (x0, x1), (y0, y1), zmax = PLACE_REGION
    p = carried[-1]
    place_dist = np.sqrt(max(x0 - p[0], 0, p[0] - x1) ** 2 + max(y0 - p[1], 0, p[1] - y1) ** 2 + max(p[2] - zmax, 0) ** 2)
    return True, bool(place_dist < PLACE_TOL)


@torch.no_grad()
def evaluate(sampler, params, model, norm, tasks, scenario, post_slide=False, seed=42, tau=1e-3, **problem_kw):
    """Run one method on the 30 tasks of one scenario; returns per-task records."""
    task = planning_task("shelf")
    robot, cost = task.robot, scenario_cost(task, scenario)
    boxes = box_groups(task.df_collision_objects) + box_groups(cost.extra_field)
    cube = torch.tensor(SCENARIOS[scenario][0])
    torch.manual_seed(seed)
    out = []
    for q_s, feasible in zip(tasks["q_start"], tasks["endpoints_feasible"]):
        def make_problem(q):
            return ShelfProblem(model, norm, task, cost, q, cube, boxes, tau=tau, **problem_kw)

        t0 = time.perf_counter()
        post = (lambda p, x: p.slide(x)) if post_slide else None
        r = rollout_chunks(make_problem, sampler, params, q_s, BATCH, N_EXEC, MAX_PLANS, max_dq=MAX_DQ, post=post)
        full = cost(r["q_traj"])
        b = int(full.argmin())
        picked, placed = pick_and_place(robot, r["q_traj"][b], SCENARIOS[scenario][0])
        free = float(full[b]) < tau
        out.append(dict(endpoints_feasible=bool(feasible), safe=free and picked and placed, goal=picked and placed,
                        collision=not free, nfe=r["nfe"],
                        n_history=[i.get("n_history", []) for i in r["infos"]],
                        fallback=sum(bool(i.get("fallback", False)) for i in r["infos"]),
                        fallback_flags=[bool(i.get("fallback", False)) for i in r["infos"]],
                        q_traj=r["q_traj"],
                        x=r["chunks"].reshape(-1, *r["chunks"].shape[2:]),
                        obs=r["obs"].reshape(-1, r["obs"].shape[-1]), wall_s=time.perf_counter() - t0))
    return out


def evaluate_unit(unit, sampler, params, model, norm, assets, post_slide=False, n_tasks=None, **problem_kw):
    tasks = {k: v[:n_tasks] for k, v in torch.load(f"{assets}/tasks/shelf.pt")["scenarios"][unit].items()}
    return evaluate(sampler, params, model, norm, tasks, unit, post_slide=post_slide, **problem_kw)


def summarize(records):
    mean = lambda k: float(sum(float(r[k]) for r in records) / len(records))  # noqa: E731
    return dict(success=mean("safe"), goal=mean("goal"), collision=mean("collision"), nfe=mean("nfe"),
                fallback_rate=sum(r["fallback"] for r in records) / sum(len(r["n_history"]) for r in records),
                n=len(records))
