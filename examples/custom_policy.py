"""Use CS-Pi with your own policy and your own constraint, with or without the YAML runner.

1. In Python (no assets, no benchmark): a toy flow whose data distribution is N(mu, s^2 I) over
   (16, 2) chunks, and a constraint "every coordinate <= 0.5" written in physical units.

       python examples/custom_policy.py

2. In an experiment YAML: `my_policy_loader` below is referenced as
   `policy: {type: "examples.custom_policy:my_policy_loader", checkpoint: avoid2d}` in
   configs/examples/custom_policy_avoid2d.yaml, which runs it on the 2D Avoidance benchmark:

       python -m cspi.core.run configs/examples/custom_policy_avoid2d.yaml
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cspi import ActionSpaceConstraint, FlowPolicy, cspi  # noqa: E402
from cspi.policies.flow import load_prior  # noqa: E402


def gaussian_velocity(mu, s):
    """Exact rectified-flow velocity field for data ~ N(mu, s^2 I): v = E[x1 - eps | x_t]."""
    def v(x, t, obs):
        t = t.view(-1, 1, 1)
        var = (1 - t) ** 2 + t ** 2 * s ** 2
        r = x - t * mu
        return mu + (t * s ** 2 - (1 - t)) * r / var
    return v


class UpperBound(ActionSpaceConstraint):
    """Every coordinate of the chunk stays <= bound. P clips (the exact Euclidean projection)."""

    def __init__(self, policy, bound=0.5):
        super().__init__(policy)
        self.bound = bound

    def action_violation(self, a):
        return torch.relu(a - self.bound).flatten(1).sum(1)

    def action_project(self, a, final=False):
        return a.clamp(max=self.bound)


def my_policy_loader(checkpoint, device="cpu"):
    """A policy loader as the YAML runner calls it: returns an *unbound* Policy. Here it wraps a
    released network as a plain function, the way you would wrap any other velocity model."""
    path = checkpoint if os.path.exists(checkpoint) else os.path.join(
        os.environ.get("CSPI_ASSETS", "assets"), "checkpoints", f"{checkpoint}.pt")
    model, norm, _ = load_prior(path, device)
    return FlowPolicy(lambda x, t, obs: model(x, t, obs), model.horizon, model.action_dim, norm)


if __name__ == "__main__":
    torch.manual_seed(0)
    unbound = FlowPolicy(gaussian_velocity(mu=1.0, s=0.3), horizon=16, action_dim=2)
    policy = unbound.condition(torch.zeros(8, 1))           # 8 samples; this toy ignores obs
    constraint = UpperBound(policy, bound=0.5)
    x, info = cspi(policy, constraint, k=1.0, k_max=5, tau=1e-6, step_size=0.1)
    print(f"max coordinate {x.max():.3f} (bound 0.5), violation {constraint.violation(x).max():.2e}, "
          f"NFE {info['nfe']}, rollback depths {[round(n, 3) for n in info['n_history']]}")
