"""Checks of the policy / constraint contracts and the YAML machinery that need no assets.

    python -m pytest tests/test_api.py -q
"""
import glob
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples"))

from custom_policy import UpperBound, gaussian_velocity  # noqa: E402

from cspi import DDIMPolicy, FlowPolicy, IdentityConstraint, cspi  # noqa: E402
from cspi.core import registry  # noqa: E402
from cspi.core.run import build_constraint, expand, load_config  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _flow(batch=16):
    return FlowPolicy(gaussian_velocity(1.0, 0.3), horizon=4, action_dim=2).condition(torch.zeros(batch, 1))


def _ddim(batch=16, mu=1.0, s=0.3, T=1000):
    """Exact epsilon prediction for data ~ N(mu, s^2 I) under a linear-beta DDPM schedule."""
    abar = torch.cumprod(1 - torch.linspace(1e-4, 0.02, T), 0)

    def eps_fn(x, i, obs):
        a = abar[i].view(-1, 1, 1)
        return (1 - a).sqrt() * (x - a.sqrt() * mu) / (a * s ** 2 + 1 - a)
    return DDIMPolicy(eps_fn, abar, horizon=4, action_dim=2).condition(torch.zeros(batch, 1))


@pytest.mark.parametrize("make", [_flow, _ddim])
def test_policy_samples_the_data(make):
    torch.manual_seed(0)
    p = make(batch=512)
    x = p.sample(step_size=0.02)
    assert abs(float(x.mean()) - 1.0) < 0.05 and abs(float(x.std()) - 0.3) < 0.05
    assert p.nfe == 50
    assert torch.allclose(p.renoise(x, 1.0, torch.randn_like(x)), x)


@pytest.mark.parametrize("make", [_flow, _ddim])
def test_cspi_satisfies_constraint(make):
    torch.manual_seed(0)
    p = make()
    c = UpperBound(p, bound=0.5)
    assert float(c.violation(p.sample(0.1)).max()) > 0  # the prior alone violates it
    x, info = cspi(p, c, k=1.0, k_max=5, tau=1e-6, step_size=0.1)
    assert float(c.violation(x).max()) <= 1e-6
    assert info["nfe"] > 10 and len(info["n_history"]) == info["n_iter"]


def test_flow_only_baselines_reject_black_box_policies():
    p = _ddim()
    with pytest.raises(NotImplementedError):
        registry.sampler("guidance")(p, UpperBound(p), alpha=1.0)


def test_identity_constraint_runs_k_max_passes():
    torch.manual_seed(0)
    p = _flow()
    x, info = cspi(p, IdentityConstraint(p), k=1.0, k_max=3, tau=0.0, n_schedule=0.5, step_size=0.1)
    assert info["n_iter"] == 3 and info["n_history"] == [0.5, 0.5, 0.5]
    assert info["nfe"] == 10 + 3 * 5  # full sample + three re-denoising passes from n = 0.5


def test_registry_resolves_names_and_import_strings():
    assert registry.sampler("cspi") is cspi
    assert registry.constraint("examples.custom_policy:UpperBound").__name__ == "UpperBound"
    assert registry.constraint("spline_collision").__name__ == "SplineCollision"
    with pytest.raises(KeyError):
        registry.sampler("nope")


def test_method_overrides_and_sweeps():
    cfg = dict(name="t", units=["a", "b"], constraint={"type": "identity"},
               methods={"M": {"sampler": "cspi", "k": [1.0, 2.0], "constraint": {"margin": [0.1, 0.2]},
                              "eval": {"seed": 3}, "post_project": True}})
    cells = expand(cfg)
    assert len(cells) == 8
    variant, label, params, unit = cells[0]
    assert variant == "M_k1.0_margin0.1" and params["constraint"] == {"margin": 0.1}
    assert params["eval"] == {"seed": 3} and params["post_project"] is True and "constraint.margin" not in params
    make = build_constraint(dict(cfg, constraint={"type": "examples.custom_policy:UpperBound", "bound": 0.0}),
                            {"bound": 0.25})
    assert make(_flow()).bound == 0.25


@pytest.mark.parametrize("path", sorted(glob.glob(os.path.join(ROOT, "configs", "**", "*.yaml"), recursive=True)))
def test_configs_are_well_formed(path):
    if os.sep + "train" + os.sep in path:
        return
    cfg = load_config(path)
    assert {"benchmark", "policy", "constraint", "units", "methods"} <= set(cfg), path
    for _, _, params, _ in expand(cfg):
        assert callable(registry.sampler(params["sampler"]))
        spec = dict(cfg["constraint"], **params["constraint"])
        assert spec["type"] in registry.CONSTRAINTS or ":" in spec["type"]
