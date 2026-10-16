"""2D Avoidance (D3IL-style): a point robot follows a flow policy of 16-step velocity
chunks from (0.1, 0.5) to (0.9, 0.5) through six training obstacles; at test time
new obstacles block some of the demonstrated routes.

Policy     : FlowMLP on (16, 2) action chunks, obs = (x, y, goal_x, goal_y), mean/std normalized.
Rollout    : receding horizon, execute 4 actions per plan, <= 400 steps, goal radius 0.10.
Constraint : chosen in the YAML (obstacles_2d in the paper: waypoints stay outside the *test*
             obstacles; identity for the SDEdit check; coupled_2d for the soft + hard composition).
Context    : every constraint is built as cls(policy, pos=pos (B, 2), obstacles=[(cx, cy, r), ...]).
Metrics    : Safe-SR (goal reached, no test-obstacle contact), fraction of rollouts whose
             waypoints stay fully inside the dilated demonstration support, NFE, logL.
"""
import time

import numpy as np
import torch

from cspi.constraints.obstacles_2d import obstacle_tensors, waypoints  # noqa: F401  (waypoints: used by figures)

START, GOAL = (0.1, 0.5), (0.9, 0.5)
GOAL_TOL, MAX_STEPS, N_EXEC = 0.10, 400, 4

TRAIN_OBSTACLES = [(0.25, 0.35, 0.06), (0.40, 0.60, 0.06), (0.55, 0.35, 0.06),
                   (0.70, 0.60, 0.06), (0.35, 0.80, 0.06), (0.65, 0.20, 0.06)]

SCENARIOS = {  # test-time obstacles (cx, cy, r)
    "left": [(0.30, 0.50, 0.08), (0.20, 0.65, 0.07)],
    "right": [(0.70, 0.40, 0.08), (0.80, 0.55, 0.07)],
    "both": [(0.30, 0.50, 0.07), (0.20, 0.65, 0.06), (0.70, 0.40, 0.07), (0.80, 0.55, 0.06)],
    "cluttered": [(0.20, 0.45, 0.06), (0.30, 0.55, 0.06), (0.45, 0.45, 0.05), (0.55, 0.55, 0.05),
                  (0.70, 0.45, 0.06), (0.80, 0.55, 0.06), (0.50, 0.75, 0.05), (0.70, 0.75, 0.06),
                  (0.45, 0.30, 0.05)],
    "narrow_passage": [(x, y, 0.18) for y in (0.70, 0.30) for x in (0.30, 0.50, 0.70)],
    "narrower_passage": [(x, y, 0.195) for y in (0.70, 0.30) for x in (0.20, 0.50, 0.80)],
    "unconstrained": [],
}
PAPER_SCENARIOS = ["left", "right", "both", "cluttered", "narrow_passage", "narrower_passage"]


class SupportMask:
    """Dilated raster of the demonstration waypoints (512x512 over [0, 1]^2, eps = 0.02)."""

    def __init__(self, path):
        d = np.load(path)
        self.mask, self.res = d["mask"].astype(bool), int(d["resolution"])

    def fraction(self, traj):
        idx = np.clip(np.floor(np.asarray(traj) * self.res).astype(int), 0, self.res - 1)
        return float(self.mask[idx[:, 1], idx[:, 0]].mean())


@torch.no_grad()
def rollout(sampler, params, policy, make_constraint, scenario, support=None):
    """Closed-loop episode; returns per-episode metrics plus the sampled chunks x (for logL scoring)."""
    test = SCENARIOS[scenario]
    centers, radii = obstacle_tensors(test)
    pos = torch.tensor([START])
    traj, chunks, obs_ns, infos = [pos[0].clone()], [], [], []
    nfe, reached, step = 0, False, 0
    while not reached and step < MAX_STEPS:
        p = policy.condition(torch.cat([pos, torch.tensor([GOAL], device=pos.device)], -1))
        x, info = sampler(p, make_constraint(p, pos=pos, obstacles=test), **params)
        nfe += info["nfe"]
        infos.append(info)
        chunks.append(x[0].clone())
        obs_ns.append(p.obs_n[0].clone())
        for a in p.normalizer.unnormalize(x)[0, :N_EXEC]:
            pos = (pos + a).clamp(0.0, 1.0)
            traj.append(pos[0].clone())
            step += 1
            if float(torch.norm(pos[0] - torch.tensor(GOAL))) < GOAL_TOL:
                reached = True
                break
            if step >= MAX_STEPS:
                break
    traj = torch.stack(traj)
    collision = bool(len(test)) and bool((torch.norm(traj[:, None] - centers[None], dim=-1) < radii).any())
    return dict(success=reached and not collision, reached_goal=reached, collision=collision, nfe=nfe,
                in_support=support.fraction(traj.numpy()) if support is not None else float("nan"),
                n_iter=[i.get("n_iter", 0) for i in infos], n_history=[i.get("n_history", []) for i in infos],
                fallback=sum(bool(i.get("fallback", False)) for i in infos), n_plans=len(infos),
                fallback_flags=[bool(i.get("fallback", False)) for i in infos],
                traj=traj.numpy(), x=torch.stack(chunks), obs=torch.stack(obs_ns))


def evaluate(sampler, params, policy, make_constraint, scenario, support=None, n_seeds=40, n_episodes=5):
    """n_seeds x n_episodes closed-loop rollouts (torch seed 1000 * s before each seed's episodes)."""
    out = []
    for s in range(n_seeds):
        torch.manual_seed(s * 1000)
        np.random.seed(s * 1000)
        for _ in range(n_episodes):
            t0 = time.perf_counter()
            out.append(rollout(sampler, params, policy, make_constraint, scenario, support))
            out[-1]["wall_s"] = time.perf_counter() - t0
    return out


@torch.no_grad()
def evaluate_coupled(sampler, params, policy, make_constraint, n_episodes=50, seed=0):
    """Soft + hard composition: n_episodes chunks from the start of `cluttered`, one batch, with every
    obstacle as context. The constraint must have a `y_ref` (coupled_2d). A chunk passes if it touches
    no obstacle and its mean |y - y_ref| is below 0.05."""
    torch.manual_seed(seed)
    obstacles = TRAIN_OBSTACLES + SCENARIOS["cluttered"]
    pos = torch.tensor([START]).expand(n_episodes, 2)
    p = policy.condition(torch.cat([pos, torch.tensor([GOAL]).expand(n_episodes, 2)], -1))
    c = make_constraint(p, pos=pos, obstacles=obstacles)
    x, info = sampler(p, c, **params)
    wp = waypoints(p.normalizer.unnormalize(x), pos)
    centers, radii = obstacle_tensors(obstacles)
    free = ~(torch.norm(wp.unsqueeze(2) - centers[None, None], dim=-1) < radii).any(-1).any(-1)
    near = (wp[..., 1] - c.y_ref).abs().mean(-1) < 0.05
    return [dict(success=bool(f and n), collision_free=bool(f), preference=bool(n), nfe=info["nfe"],
                 n_history=info.get("n_history", []), x=x[i:i + 1], obs=p.obs_n[i:i + 1])
            for i, (f, n) in enumerate(zip(free.tolist(), near.tolist()))]


def training_chunks(demos_path, horizon=16):
    """All length-`horizon` windows of the demonstrations -> (actions (N, H, 2), obs (N, 4))."""
    d = torch.load(demos_path)
    obs, act = d["obs"], d["action"]
    X, O = [], []
    for o, a in zip(obs, act):
        for s in range(a.shape[0] - horizon + 1):
            X.append(a[s:s + horizon, :2])
            O.append(o[s, :4])
    return torch.stack(X), torch.stack(O)


def _mean(records, key):
    return float(np.mean([float(r[key]) for r in records]))


def evaluate_unit(unit, sampler, params, policy, make_constraint, assets, n_seeds=40, n_episodes=5):
    """unit: a scenario name, or "coupled_s<seed>" for the soft + hard composition experiment
    (n_episodes chunks from the start of `cluttered`)."""
    if unit.startswith("coupled"):
        return evaluate_coupled(sampler, params, policy, make_constraint, n_episodes, int(unit.split("_s")[-1]))
    support = SupportMask(f"{assets}/data/avoid2d_support.npz")
    return evaluate(sampler, params, policy, make_constraint, unit, support, n_seeds, n_episodes)


def summarize(records):
    m = dict(success=_mean(records, "success"), nfe=_mean(records, "nfe"), n=len(records))
    if "in_support" in records[0]:
        m.update(fully_in_support=float(np.mean([r["in_support"] == 1.0 for r in records])),
                 in_support=_mean(records, "in_support"),
                 goal=_mean(records, "reached_goal"), collision=_mean(records, "collision"),
                 fallback_rate=sum(r["fallback"] for r in records) / max(sum(r["n_plans"] for r in records), 1))
    else:
        m.update(collision_free=_mean(records, "collision_free"), preference=_mean(records, "preference"))
    return m
