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

from cspi import cbf, samplers
from cspi.flow import load_prior

ASSETS = os.environ.get("CSPI_ASSETS", "assets")
SAMPLERS = dict(samplers.SAMPLERS, **cbf.SAMPLERS)
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
    b, pk = case["benchmark"], dict(case["problem"])
    model, norm, _ = load_prior(os.path.join(ASSETS, "checkpoints", f"{b}.pt"))
    post = pk.pop("post_slide", False)
    if b == "avoid2d":
        from cspi.benchmarks import avoid2d as B
        p = B.AvoidProblem(model, norm, torch.tensor([case["pos"]]), B.SCENARIOS[case["scenario"]])
        decode = p.actions
    elif b == "spheres":
        from cspi.benchmarks import spheres as B
        tasks = torch.load(os.path.join(ASSETS, "tasks", "spheres.pt"))
        task, cost = B.load(case["split"])
        p = B.SplineProblem(model, norm, task, cost, tasks["q_start"][case["task"]], tasks["q_goal"][case["task"]])
        decode = norm.unnormalize
    elif b == "orient":
        from cspi.benchmarks import orient as B
        from cspi.benchmarks.mpd import planning_task
        tasks = torch.load(os.path.join(ASSETS, "tasks", "orient.pt"))
        qs, qg = tasks["q_start"][case["task"]], tasks["q_goal"][case["task"]]
        p = B.OrientProblem(model, norm, planning_task("spheres").robot, qs.view(1, -1), qg, **pk)
        decode = p.joints
    else:
        from cspi.benchmarks import shelf as B
        from cspi.benchmarks.mpd import planning_task
        task = planning_task("shelf")
        cost = B.scenario_cost(task, case["scenario"])
        q = torch.load(os.path.join(ASSETS, "tasks", "shelf.pt"))["scenarios"][case["scenario"]]["q_start"][case["task"]]
        boxes = B.box_groups(task.df_collision_objects) + B.box_groups(cost.extra_field)
        p = B.ShelfProblem(model, norm, task, cost, q.view(1, -1).expand(4, -1).contiguous(),
                           torch.tensor(B.SCENARIOS[case["scenario"]][0]), boxes, **pk)
        decode = p.joints
    torch.manual_seed(case["seed"])
    x, info = SAMPLERS[case["sampler"]](p, **case["params"])
    if post:
        x = p.slide(x)
    return decode(x), info["nfe"]


# Orient and Shelf project with a strong-Wolfe L-BFGS line search, which turns last-bit floating-point
# differences between CPUs into up to ~5e-3 rad; NFE must still match exactly.
ATOL = dict(orient=1e-2, shelf=1e-2)


@pytest.mark.parametrize("case", GOLDEN, ids=[f"{c['benchmark']}-{c['sampler']}-{c['seed']}" for c in GOLDEN])
def test_matches_paper(case):
    out, nfe = _replay(case)
    assert nfe == case["nfe"]
    atol = ATOL.get(case["benchmark"], 1e-4)
    assert torch.allclose(out, case["output"], atol=atol), (out - case["output"]).abs().max()
