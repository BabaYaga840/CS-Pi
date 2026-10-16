"""Policies: the contract CS-Pi samples from (base.Policy) and its implementations.

POLICIES maps the `policy.type` names usable in an experiment YAML to loader functions; a loader
takes the remaining keys of the `policy:` block (plus `device`) and returns an unbound Policy.
Any other loader can be named as "package.module:function".
"""
from cspi.policies.base import Policy
from cspi.policies.diffusion import DDIMPolicy
from cspi.policies.flow import FlowPolicy, load_flow_policy
from cspi.policies.normalizer import Normalizer

POLICIES = dict(flow=load_flow_policy)

__all__ = ["Policy", "FlowPolicy", "DDIMPolicy", "Normalizer", "load_flow_policy", "POLICIES"]
