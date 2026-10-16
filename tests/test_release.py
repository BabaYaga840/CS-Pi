"""Checks for the release.

test_qp_solver     : the CBF QP solver against scipy on random feasible projection QPs (no assets needed).
test_matches_paper : assets/golden.pt holds outputs of the original research code that produced the paper's
                     numbers, for single sampler calls with fixed seeds on every benchmark. Each case is
                     replayed with this code and must match. Panda cases need MPD (see README).

    python -m pytest tests/ -q
"""
import os

import numpy as np
import pytest
import torch

from cspi.baselines import cbf
from cspi.core import registry
from cspi.policies.flow import load_flow_policy

ASSETS = os.environ.get("CSPI_ASSETS", "assets")
GOLDEN = torch.load(os.path.join(ASSETS, "golden.pt")) if os.path.exists(os.path.join(ASSETS, "golden.pt")) else []


def test_qp_solver():
    from scipy.optimize import minimize
    rng = np.random.default_rng(0)
    for trial in range(100):
        n, m = rng.integers(2, 20), rng.integers(1, 30)
        A, v = rng.normal(size=(m, n)), rng.normal(size=n)
        c = A @ (2 * rng.normal(size=n)) - np.abs(rng.normal(size=m))
        u = cbf.solve_qp(v, A, c)
        ref = minimize(lambda z: ((z - v) ** 2).sum(), v, method="SLSQP", options={"maxiter": 2000, "ftol": 1e-14},
                       constraints=[{"type": "ineq", "fun": lambda z: A @ z - c}])
        if ref.success:
            assert np.abs(u - ref.x).max() < 1e-5


def _replay(case):
    """Rebuild the golden case's (policy, constraint) pair with the release API and rerun its sampler."""
    b, pk = case["benchmark"], dict(case["problem"])
    policy = load_flow_policy(os.path.join(ASSETS, "checkpoints", f"{b}.pt"))
    post = pk.pop("post_slide", False)
    if b == "avoid2d":
        from cspi.benchmarks import avoid2d as B
        from cspi.constraints.obstacles_2d import Obstacles2D
        pos = torch.tensor([case["pos"]])
        p = policy.condition(torch.cat([pos, torch.tensor([B.GOAL])], -1))
        c = Obstacles2D(p, pos, B.SCENARIOS[case["scenario"]])
    elif b == "spheres":
        from cspi.benchmarks import spheres as B
        from cspi.constraints.spline_collision import SplineCollision
        tasks = torch.load(os.path.join(ASSETS, "tasks", "spheres.pt"))
        task, cost = B.load(case["split"])
        qs, qg = tasks["q_start"][case["task"]], tasks["q_goal"][case["task"]]
        p = policy.condition(torch.cat([qs, qg]).view(1, -1).expand(8, -1))
        c = SplineCollision(p, task, cost, qs, qg)
    elif b == "orient":
        from cspi.benchmarks.mpd import planning_task
        from cspi.constraints.orientation import OrientationLock
        tasks = torch.load(os.path.join(ASSETS, "tasks", "orient.pt"))
        qs, qg = tasks["q_start"][case["task"]].view(1, -1), tasks["q_goal"][case["task"]].view(1, -1)
        p = policy.condition(torch.cat([qs, qg], -1))
        c = OrientationLock(p, planning_task("spheres").robot, qs, **pk)
    else:
        from cspi.benchmarks import shelf as B
        from cspi.benchmarks.mpd import planning_task
        from cspi.constraints.chunk_collision import ChunkCollision
        task = planning_task("shelf")
        cost = B.scenario_cost(task, case["scenario"])
        q = torch.load(os.path.join(ASSETS, "tasks", "shelf.pt"))["scenarios"][case["scenario"]]["q_start"][case["task"]]
        q = q.view(1, -1).expand(4, -1).contiguous()
        cube = torch.tensor(B.SCENARIOS[case["scenario"]][0])
        boxes = B.box_groups(task.df_collision_objects) + B.box_groups(cost.extra_field)
        p = policy.condition(torch.cat([q, cube.view(1, -1).expand(4, -1)], -1))
        c = ChunkCollision(p, task, cost, q, boxes, **pk)
    torch.manual_seed(case["seed"])
    x, info = registry.sampler(case["sampler"])(p, c, **case["params"])
    if post:
        x = c.post_project(x)
    return p.normalizer.unnormalize(x), info["nfe"]


# Orient and Shelf project with a strong-Wolfe L-BFGS line search, which turns last-bit floating-point
# differences between CPUs into up to ~5e-3 rad; NFE must still match exactly.
ATOL = dict(orient=1e-2, shelf=1e-2)


@pytest.mark.parametrize("case", GOLDEN, ids=[f"{c['benchmark']}-{c['sampler']}-{c['seed']}" for c in GOLDEN])
def test_matches_paper(case):
    out, nfe = _replay(case)
    assert nfe == case["nfe"]
    atol = ATOL.get(case["benchmark"], 1e-4)
    assert torch.allclose(out, case["output"], atol=atol), (out - case["output"]).abs().max()
