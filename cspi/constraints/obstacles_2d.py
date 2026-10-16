"""2D obstacle avoidance for chunks of velocity actions (the 2D Avoidance benchmark).

A chunk a (B, H, 2) executed from pos (B, 2) visits the waypoints pos + cumsum(a).

Obstacles2D  : waypoints stay outside circular obstacles (plus a margin).
               P pushes every penetrating waypoint radially onto the obstacle surface.
Coupled2D    : the same hard constraint plus a soft preference to keep y near y_ref;
               P = obstacle projection after the proximal step of the preference.

Context supplied by the benchmark: pos (B, 2), obstacles [(cx, cy, r), ...].
"""
import torch

from cspi.constraints.base import ActionSpaceConstraint


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


class Obstacles2D(ActionSpaceConstraint):
    """margin: inflation of every obstacle for violation and projection.
    guide_temperature, guide_margin: shape of the sigmoid cost used by guidance / Trust."""

    def __init__(self, policy, pos, obstacles, margin=0.01, guide_temperature=0.02, guide_margin=0.005):
        super().__init__(policy)
        self.pos = pos
        self.centers, self.radii = obstacle_tensors(obstacles, pos.device)
        self.margin, self.guide_temperature, self.guide_margin = margin, guide_temperature, guide_margin

    def action_violation(self, a):
        return penetration(waypoints(a, self.pos), self.centers, self.radii, self.margin)

    def action_project(self, a, final=False):
        return project_actions(a, self.pos, self.centers, self.radii, self.margin)

    def action_cost(self, a):
        return soft_obstacle_cost(waypoints(a, self.pos), self.centers, self.radii, self.guide_temperature,
                                  self.guide_margin)

    def barrier_rows(self, x, cbf_margin=0.005):
        """b_{k,m} = |p_k - c_m| - (r_m + 0.005); p_k depends on actions 0..k (rows chained by cumsum)."""
        B, H, _ = x.shape
        diff = waypoints(self.actions(x), self.pos).unsqueeze(2) - self.centers[None, None]
        dist = diff.norm(dim=-1)
        grad = torch.where(dist.unsqueeze(-1) > 1e-12, diff / dist.clamp(min=1e-12).unsqueeze(-1),
                           torch.tensor([1.0, 0.0]))
        A = torch.tril(torch.ones(H, H))[None, :, None, :, None] * (grad * self.norm.x_b).unsqueeze(3)
        return (dist - (self.radii + cbf_margin)).reshape(B, -1), A.reshape(B, -1, H * 2)


class Coupled2D(Obstacles2D):
    """Soft + hard composition (Fig. softhard-2d): avoid all obstacles (hard) and keep the chunk's
    y-coordinate near y_ref (soft, L = mean_t (y_t - y_ref)^2, weight lam).

    P = obstacle projection after the proximal step of the preference, prox_{lam L};
    the terminal fallback only enforces the hard constraint.
    """

    def __init__(self, policy, pos, obstacles, y_ref=0.7, lam=1.0, margin=0.005):
        super().__init__(policy, pos, obstacles, margin=margin, guide_margin=margin)
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

    def action_violation(self, a):
        return self.preference(a) + penetration(waypoints(a, self.pos), self.centers, self.radii, self.margin)

    def action_project(self, a, final=False):
        return project_actions(self.prox(a), self.pos, self.centers, self.radii, self.margin)

    def fallback(self, x):
        return self.norm.normalize(Obstacles2D.action_project(self, self.actions(x)))

    def action_cost(self, a):
        return super().action_cost(a) + self.lam * self.preference(a)
