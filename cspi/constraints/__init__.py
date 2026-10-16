"""Constraints and their projections (base.Constraint is the contract).

CONSTRAINTS maps the `constraint.type` names usable in an experiment YAML to classes, built as
cls(bound_policy, **benchmark_context, **yaml_params). Any other class can be named as
"package.module:Class". The Panda classes are imported lazily (they need MPD objects at runtime,
but not MPD itself at import time).
"""
from cspi.constraints.base import ActionSpaceConstraint, Constraint, IdentityConstraint
from cspi.constraints.obstacles_2d import Coupled2D, Obstacles2D

CONSTRAINTS = {
    "identity": IdentityConstraint,
    "obstacles_2d": Obstacles2D,
    "coupled_2d": Coupled2D,
    "spline_collision": "cspi.constraints.spline_collision:SplineCollision",
    "orientation_lock": "cspi.constraints.orientation:OrientationLock",
    "chunk_collision": "cspi.constraints.chunk_collision:ChunkCollision",
}

__all__ = ["Constraint", "ActionSpaceConstraint", "IdentityConstraint", "Obstacles2D", "Coupled2D", "CONSTRAINTS"]
