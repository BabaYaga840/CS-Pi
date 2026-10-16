"""Name -> object resolution for experiment YAMLs.

Every `type` / `sampler` / `benchmark` field in a YAML is either a short name registered below or a
"package.module:attribute" string, so new policies, constraints, samplers and benchmarks plug in
without editing this package.
"""
import importlib

from cspi.baselines import BASELINES
from cspi.constraints import CONSTRAINTS
from cspi.core.algorithm import cspi
from cspi.policies import POLICIES

SAMPLERS = dict(cspi=cspi, **BASELINES)


def import_string(path):
    """'package.module:attr' -> the attribute."""
    module, _, attr = path.partition(":")
    obj = importlib.import_module(module)
    for a in attr.split("."):
        obj = getattr(obj, a)
    return obj


def resolve(name, registry, kind):
    if ":" in name:
        return import_string(name)
    if name not in registry:
        raise KeyError(f"unknown {kind} '{name}'; registered: {sorted(registry)}, or give 'package.module:attr'")
    obj = registry[name]
    return import_string(obj) if isinstance(obj, str) else obj


def sampler(name):
    return resolve(name, SAMPLERS, "sampler")


def constraint(name):
    return resolve(name, CONSTRAINTS, "constraint")


def policy_loader(name):
    return resolve(name, POLICIES, "policy")


def benchmark(name):
    """A module in cspi/benchmarks/, or any importable module path ('my_pkg.my_bench')."""
    return importlib.import_module(name if "." in name else f"cspi.benchmarks.{name}")


def with_post_projection(fn):
    """Sampler that applies constraint.post_project to the output of `fn` (method key post_project: true)."""
    def wrapped(policy, constraint, **kw):
        x, info = fn(policy, constraint, **kw)
        return constraint.post_project(x), info
    return wrapped
