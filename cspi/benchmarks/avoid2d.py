"""2D Avoidance (D3IL-style): a point robot follows a flow policy of 16-step velocity
chunks from (0.1, 0.5) to (0.9, 0.5) through six training obstacles; at test time
new obstacles block some of the demonstrated routes.

Policy     : FlowMLP on (16, 2) action chunks, obs = (x, y, goal_x, goal_y), mean/std normalized.
Rollout    : receding horizon, execute 4 actions per plan, <= 400 steps, goal radius 0.10.
Constraint : waypoints of the chunk stay outside the *test* obstacles (plus a margin).
P          : push every penetrating waypoint radially onto the obstacle surface.
Metrics    : Safe-SR (goal reached, no test-obstacle contact), fraction of rollouts whose
             waypoints stay fully inside the dilated demonstration support, NFE, logL.
"""
import numpy as np
import torch

from cspi.samplers import Problem

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


def obstacle_tensors(obstacles, device="cpu"):
    t = torch.tensor(obstacles, dtype=torch.float32, device=device).reshape(-1, 3)
    return t[:, :2], t[:, 2]


def waypoints(actions, pos):
    """Absolute positions reached by a chunk of velocity actions (B, H, 2) from pos (B, 2)."""
    return pos.unsqueeze(1) + torch.cumsum(actions, dim=1)


def penetration(wp, centers, radii, margin):
    """Summed penetration depth of all waypoints into all (inflated) obstacles -> (B,)."""
    dist = torch.norm(wp.unsqueeze(2) - centers[None, None], dim=-1)
    return torch.relu(radii + margin - dist).sum(dim=(1, 2))


def project_actions(actions, pos, centers, radii, margin):
    """Move every waypoint inside an inflated obstacle radially onto its surface (3 passes,
    to resolve overlaps), then convert the waypoints back to velocity actions."""
    wp = waypoints(actions, pos)
    r = radii + margin
    for _ in range(3):
        diff = wp.unsqueeze(2) - centers[None, None]
        dist = torch.norm(diff, dim=-1)
        inside = dist < r
        if not inside.any():
            break
        surface = centers[None, None] + diff / (dist.unsqueeze(-1) + 1e-9) * r[None, None, :, None]
        for j in range(centers.shape[0]):
            wp = torch.where(inside[:, :, j:j + 1], surface[:, :, j], wp)
    out = torch.zeros_like(actions)
    out[:, 0] = wp[:, 0] - pos
    out[:, 1:] = wp[:, 1:] - wp[:, :-1]
    return out


def soft_obstacle_cost(wp, centers, radii, temperature=0.02, margin=0.005):
    """Sigmoid barrier sum_{t,j} sigmoid((r_j + margin - |p_t - c_j|) / T) -> (B,)."""
    dist = torch.norm(wp.unsqueeze(2) - centers[None, None], dim=-1)
    return torch.sigmoid((radii + margin - dist) / temperature).sum(dim=(1, 2))


class AvoidProblem(Problem):
    """One plan step: sample a chunk from the current position that avoids the test obstacles."""

    def __init__(self, model, norm, pos, obstacles, margin=0.01, guide_temperature=0.02, guide_margin=0.005):
        goal = torch.tensor([GOAL], device=pos.device).expand(pos.shape[0], 2)
        obs_n = norm.normalize_obs(torch.cat([pos, goal], -1))
        super().__init__(model, obs_n, (pos.shape[0], model.horizon, model.action_dim))
        self.norm, self.pos = norm, pos
        self.centers, self.radii = obstacle_tensors(obstacles, pos.device)
        self.margin, self.guide_temperature, self.guide_margin = margin, guide_temperature, guide_margin

    def actions(self, x):
        return self.norm.unnormalize(x)

    def violation(self, x):
        return penetration(waypoints(self.actions(x), self.pos), self.centers, self.radii, self.margin)

    def project(self, x, final=False):
        return self.norm.normalize(project_actions(self.actions(x), self.pos, self.centers, self.radii, self.margin))

    def cost(self, x):
        wp = waypoints(self.actions(x), self.pos)
        return soft_obstacle_cost(wp, self.centers, self.radii, self.guide_temperature, self.guide_margin)

    def barrier_rows(self, x, cbf_margin=0.005):
        """b_{k,m} = |p_k - c_m| - (r_m + 0.005); p_k depends on actions 0..k (rows chained by cumsum)."""
        B, H, _ = x.shape
        diff = waypoints(self.actions(x), self.pos).unsqueeze(2) - self.centers[None, None]
        dist = diff.norm(dim=-1)
        grad = torch.where(dist.unsqueeze(-1) > 1e-12, diff / dist.clamp(min=1e-12).unsqueeze(-1),
                           torch.tensor([1.0, 0.0]))
        A = torch.tril(torch.ones(H, H))[None, :, None, :, None] * (grad * self.norm.x_b).unsqueeze(3)
        return (dist - (self.radii + cbf_margin)).reshape(B, -1), A.reshape(B, -1, H * 2)


class CoupledProblem(AvoidProblem):
    """Soft + hard composition (Fig. softhard-2d): avoid *all* obstacles (hard) and keep the
    chunk's y-coordinate near y_ref (soft, L = mean_t (y_t - y_ref)^2).

    P = obstacle projection after the proximal step of the preference, prox_{lam L};
    the terminal fallback only enforces the hard constraint.
    """

    def __init__(self, model, norm, pos, obstacles, y_ref=0.7, lam=1.0, margin=0.005):
        super().__init__(model, norm, pos, obstacles, margin=margin, guide_margin=margin)
        self.y_ref, self.lam = y_ref, lam

    def preference(self, actions):
        return ((waypoints(actions, self.pos)[..., 1] - self.y_ref) ** 2).mean(-1)

    def prox(self, actions, n_iter=25):
        """argmin_a' lam L(a') + 0.5 ||a' - a||^2 over the y-velocities (L-BFGS)."""
        a0 = actions.detach()
        y = a0[..., 1].clone().requires_grad_(True)
        opt = torch.optim.LBFGS([y], lr=0.1, max_iter=n_iter, tolerance_grad=1e-7, tolerance_change=1e-9,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            a = torch.stack([a0[..., 0], y], -1)
            f = self.lam * self.preference(a).sum() + 0.5 * (y - a0[..., 1]).pow(2).sum()
            f.backward()
            return f

        with torch.enable_grad():
            opt.step(closure)
        return torch.stack([a0[..., 0], y.detach()], -1)

    def violation(self, x):
        a = self.actions(x)
        return self.preference(a) + penetration(waypoints(a, self.pos), self.centers, self.radii, self.margin)

    def project(self, x, final=False):
        a = project_actions(self.prox(self.actions(x)), self.pos, self.centers, self.radii, self.margin)
        return self.norm.normalize(a)

    def fallback(self, x):
        return super().project(x)

    def cost(self, x):
        a = self.actions(x)
        return super().cost(x) + self.lam * self.preference(a)


class SelfAnchorProblem(AvoidProblem):
    """SDEdit mechanism check: CS-Pi's renoise / re-denoise loop with P = identity (no constraint).
    Run with tau = 0, a constant n_schedule and k_max = number of passes."""

    def violation(self, x):
        return torch.ones(x.shape[0])

    def project(self, x, final=False):
        return x

    def fallback(self, x):
        return x


class SupportMask:
    """Dilated raster of the demonstration waypoints (512x512 over [0, 1]^2, eps = 0.02)."""

    def __init__(self, path):
        d = np.load(path)
        self.mask, self.res = d["mask"].astype(bool), int(d["resolution"])

    def fraction(self, traj):
        idx = np.clip(np.floor(np.asarray(traj) * self.res).astype(int), 0, self.res - 1)
        return float(self.mask[idx[:, 1], idx[:, 0]].mean())


@torch.no_grad()
def rollout(sampler, params, model, norm, scenario, support=None, margin=0.01, problem=AvoidProblem):
    """Closed-loop episode; returns per-episode metrics plus the sampled chunks x (for logL scoring)."""
    test = SCENARIOS[scenario]
    centers, radii = obstacle_tensors(test)
    pos = torch.tensor([START])
    traj, chunks, obs_ns, infos = [pos[0].clone()], [], [], []
    nfe, reached, step = 0, False, 0
    while not reached and step < MAX_STEPS:
        p = problem(model, norm, pos, test, margin=margin)
        x, info = sampler(p, **params)
        nfe += info["nfe"]
        infos.append(info)
        chunks.append(x[0].clone())
        obs_ns.append(p.obs_n[0].clone())
        for a in p.actions(x)[0, :N_EXEC]:
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


def evaluate(sampler, params, model, norm, scenario, support=None, n_seeds=40, n_episodes=5, margin=0.01,
             problem=AvoidProblem):
    """n_seeds x n_episodes closed-loop rollouts (torch seed 1000 * s before each seed's episodes)."""
    out = []
    for s in range(n_seeds):
        torch.manual_seed(s * 1000)
        np.random.seed(s * 1000)
        for _ in range(n_episodes):
            out.append(rollout(sampler, params, model, norm, scenario, support, margin, problem))
    return out


@torch.no_grad()
def evaluate_coupled(sampler, params, model, norm, n_episodes=50, seed=0, y_ref=0.7, lam=1.0):
    """Soft + hard composition: n_episodes chunks from the start of `cluttered`, one batch.
    A chunk passes if it touches no obstacle and its mean |y - y_ref| is below 0.05."""
    torch.manual_seed(seed)
    obstacles = TRAIN_OBSTACLES + SCENARIOS["cluttered"]
    pos = torch.tensor([START]).expand(n_episodes, 2)
    p = CoupledProblem(model, norm, pos, obstacles, y_ref=y_ref, lam=lam)
    x, info = sampler(p, **params)
    wp = waypoints(p.actions(x), pos)
    centers, radii = obstacle_tensors(obstacles)
    free = ~(torch.norm(wp.unsqueeze(2) - centers[None, None], dim=-1) < radii).any(-1).any(-1)
    near = (wp[..., 1] - y_ref).abs().mean(-1) < 0.05
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


def evaluate_unit(unit, sampler, params, model, norm, assets, n_seeds=40, n_episodes=5, margin=0.01,
                  self_anchor=False, y_ref=0.7, lam=1.0):
    """unit: a scenario name, or "coupled_s<seed>" for the soft + hard composition experiment
    (n_episodes chunks from the start of `cluttered`)."""
    if unit.startswith("coupled"):
        return evaluate_coupled(sampler, params, model, norm, n_episodes, int(unit.split("_s")[-1]), y_ref, lam)
    support = SupportMask(f"{assets}/data/avoid2d_support.npz")
    return evaluate(sampler, params, model, norm, unit, support, n_seeds, n_episodes, margin,
                    SelfAnchorProblem if self_anchor else AvoidProblem)


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
