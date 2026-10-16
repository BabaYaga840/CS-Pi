"""Shared environment glue for the three Franka Panda benchmarks, built on MPD (Carvalho et al.).

MPD supplies the robot (forward kinematics, collision spheres, joint limits), the
environments and their signed-distance collision fields. Install it from
https://github.com/joaoamcarvalho/mpd-splines-public at commit 3676cbf, apply
third_party/mpd.patch, and point MPD_ROOT at the checkout. Everything here runs on
CPU (the FK library keeps internal CPU tensors).
"""
import os
import sys

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


# ---------------------------------------------------------------------------
# Receding-horizon rollout of a chunk policy (Orient, Shelf)
# ---------------------------------------------------------------------------

@torch.no_grad()
def rollout_chunks(make_problem, sampler, params, q_start, batch, n_exec, max_plans,
                   q_goal=None, goal_tol=0.0, max_dq=0.0):
    """Plan a chunk from the current joint state, execute its first n_exec steps, repeat.

    make_problem(q_current (B, 7)) -> (bound policy, constraint).  A sample that gets within goal_tol
    of q_goal freezes in place; the rollout ends when all are frozen or after max_plans plans.
    max_dq > 0 rate-limits executed joints (rad per step) relative to the last executed state.
    Chunks are absolute joint configurations in the policy's normalized space.
    """
    q = q_start.view(1, -1).expand(batch, -1).clone()
    executed, frozen = [q], torch.zeros(batch, dtype=torch.bool)
    chunks, obs, infos, nfe = [], [], [], 0
    for plan in range(max_plans):
        if q_goal is not None and plan > 0:
            frozen = frozen | ((q - q_goal.view(1, -1)).norm(dim=-1) < goal_tol)
            if bool(frozen.all()):
                break
        policy, constraint = make_problem(q)
        x, info = sampler(policy, constraint, **params)
        nfe += info["nfe"]
        infos.append(info)
        chunks.append(x.detach().cpu())
        obs.append(policy.obs_n.detach().cpu())
        chunk = policy.normalizer.unnormalize(x).cpu()
        for i in range(n_exec):
            target = chunk[:, i]
            if max_dq > 0:
                target = executed[-1] + (target - executed[-1]).clamp(-max_dq, max_dq)
            executed.append(torch.where(frozen.view(-1, 1), q, target))
        q = torch.where(frozen.view(-1, 1), q, executed[-1])
    return dict(q_traj=torch.stack(executed[1:], dim=1), chunks=torch.stack(chunks), obs=torch.stack(obs),
                infos=infos, nfe=nfe, n_plans=len(infos))
