"""Baselines CS-Pi is compared against, all with the sampler signature of cspi/core/algorithm.py:

    sampler(policy, constraint, **hyperparameters) -> (x, info)

    simple.py    unguided, projection, rejection         black-box policy; constraint.violation / project
    guidance.py  gradient guidance, Trust Sampling        policy.velocity; constraint.cost (+ grad_mask)
    cbf.py       SafeDiffuser, SafeFlowMatcher            policy.velocity; constraint.barrier_rows
"""
from cspi.baselines.cbf import safediffuser, safeflowmatcher
from cspi.baselines.guidance import guidance, trust
from cspi.baselines.simple import projection, rejection, unguided

BASELINES = dict(unguided=unguided, projection=projection, rejection=rejection, guidance=guidance, trust=trust,
                 safediffuser=safediffuser, safeflowmatcher=safeflowmatcher)
