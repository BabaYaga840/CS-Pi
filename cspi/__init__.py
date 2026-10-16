"""CS-Pi: constraint-satisfying sampling from a frozen flow / diffusion policy.

    from cspi import FlowPolicy, ActionSpaceConstraint, cspi
    policy = FlowPolicy(my_velocity_fn, horizon=16, action_dim=2).condition(obs)
    x, info = cspi(policy, MyConstraint(policy, ...), k=20.0, k_max=5, tau=0.01)

See FOR_AGENTS.md for the policy and constraint contracts and every hyperparameter.
"""
from cspi.constraints.base import ActionSpaceConstraint, Constraint, IdentityConstraint
from cspi.core.algorithm import cspi
from cspi.policies.base import Policy
from cspi.policies.diffusion import DDIMPolicy
from cspi.policies.flow import FlowPolicy, load_flow_policy
from cspi.policies.normalizer import Normalizer

__all__ = ["cspi", "Policy", "FlowPolicy", "DDIMPolicy", "load_flow_policy", "Normalizer",
           "Constraint", "ActionSpaceConstraint", "IdentityConstraint"]
