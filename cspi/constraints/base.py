"""The constraint contract: what CS-Pi and the baselines need from a constraint.

A Constraint is built for one bound policy and one decision (plan step / task), and acts on samples
x in the policy's *model space* (the same space as the policy's samples). Distances CS-Pi measures,
||P(x) - x||, are therefore model-space distances.

Required by CS-Pi and Projection:

    violation(x)          -> (B,) >= 0, 0 = feasible; compared against the sampler's `tau`
    project(x, final)     -> P(x), a projection onto (or a step towards) the feasible set.
                             final=True is the terminal projection, usually with a bigger budget.

Optional:

    fallback(x)           terminal projection of CS-Pi (default: project(x, final=True))
    post_project(x)       applied after any sampler whose method sets `post_project: true`
                          (default: fallback(x))
    cost(x)               (B,) differentiable surrogate, for gradient guidance / Trust Sampling
    barrier_rows(x)       (b (B, R), A (B, R, H*d)) barrier values >= 0 when safe and their
                          Jacobian w.r.t. flattened x, for SafeDiffuser / SafeFlowMatcher
    grad_mask             (1, H, d) mask of coordinates gradient methods may move (None = all)

Benchmarks construct constraints as `cls(policy, **context, **params)`: `context` is what the
benchmark knows at that decision (current position, obstacles, task, ...), `params` come from the
experiment YAML. See the concrete classes for the context each one expects.
"""
import torch


class Constraint:

    grad_mask = None

    def __init__(self, policy):
        self.policy, self.norm = policy, policy.normalizer

    def violation(self, x):
        raise NotImplementedError

    def project(self, x, final=False):
        raise NotImplementedError

    def fallback(self, x):
        return self.project(x, final=True)

    def post_project(self, x):
        return self.fallback(x)

    def cost(self, x):
        raise NotImplementedError(f"{type(self).__name__} defines no differentiable cost (needed by guidance / Trust)")

    def barrier_rows(self, x):
        raise NotImplementedError(f"{type(self).__name__} defines no barrier rows (needed by the CBF baselines)")


class ActionSpaceConstraint(Constraint):
    """Write a constraint in physical units: subclasses implement the `action_*` methods on
    a = normalizer.unnormalize(x); the model-space interface is derived from them."""

    def actions(self, x):
        return self.norm.unnormalize(x)

    def action_violation(self, a):
        raise NotImplementedError

    def action_project(self, a, final=False):
        raise NotImplementedError

    def action_cost(self, a):
        raise NotImplementedError

    def violation(self, x):
        return self.action_violation(self.actions(x))

    def project(self, x, final=False):
        return self.norm.normalize(self.action_project(self.actions(x), final))

    def cost(self, x):
        return self.action_cost(self.actions(x))


class IdentityConstraint(Constraint):
    """No constraint and P = identity, but always "violated": CS-Pi then just renoises and re-denoises
    its own samples (the SDEdit mechanism check; run with tau = 0 and a constant n_schedule)."""

    def __init__(self, policy, **context):
        super().__init__(policy)

    def violation(self, x):
        return torch.ones(x.shape[0])

    def project(self, x, final=False):
        return x

    def fallback(self, x):
        return x
