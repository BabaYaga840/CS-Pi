"""Run the experiment described by one YAML file.

    python -m cspi.core.run configs/tab_shelf.yaml            # all cells in sequence (finished cells are skipped)
    python -m cspi.core.run configs/tab_shelf.yaml --list     # print the number of cells (size of a SLURM array)
    python -m cspi.core.run configs/tab_shelf.yaml --cell 7   # one cell

An experiment YAML chooses every component by name (see FOR_AGENTS.md for the full schema):

    benchmark:  avoid2d                                  cspi/benchmarks/<name>.py or "pkg.module"
    policy:     {type: flow, checkpoint: avoid2d}        cspi/policies: loader name + its arguments
    constraint: {type: obstacles_2d, margin: 0.01}       cspi/constraints: class name + its parameters
    units:      [left, right]                            scenarios / splits / seeds of the benchmark
    eval:       {n_seeds: 40, n_episodes: 5}             benchmark (evaluate_unit) arguments
    methods:                                             label -> sampler and hyperparameters
      CS-Pi: {sampler: cspi, k: [10.0, 20.0], k_max: 5, tau: 0.01, step_size: 0.5}
      Trust: {sampler: trust, w: 0.05, j_max: 5, post_project: true, constraint: {proj_steps: 10}}

A method may override `constraint:` and `eval:` keys. Any list-valued value is a sweep; every
(method variant, unit) pair is one cell and writes results/<name>/<variant>/<unit>.json (metrics,
including the policy's log-likelihood of the samples) and a .pt file with the rollouts.
"""
import argparse
import functools
import itertools
import json
import os
import time

import torch
import yaml

from cspi.core import registry
from cspi.loglik import log_likelihood

ASSETS = os.environ.get("CSPI_ASSETS", "assets")
RESULTS = os.environ.get("CSPI_RESULTS", "results")
SECTIONS = ("constraint", "eval")  # per-method overrides of the experiment-level blocks


def load_config(path):
    """Read a config. `methods_from: other.yaml` (optionally with `only: [labels]`) reuses the method
    definitions of another config, so derived experiments cannot drift from the paper tables."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("name", os.path.splitext(os.path.basename(path))[0])
    if "methods_from" in cfg:
        with open(os.path.join(os.path.dirname(path), cfg["methods_from"])) as f:
            methods = yaml.safe_load(f)["methods"]
        cfg["methods"] = {k: v for k, v in methods.items() if k in cfg.get("only", methods)}
    return cfg


def expand(cfg):
    """All cells of a config: (variant, method label, params, unit). params holds the sampler
    hyperparameters plus the method's `constraint` and `eval` overrides, sweeps resolved."""
    out = []
    for label, m in cfg["methods"].items():
        flat = {k: v for k, v in m.items() if k not in SECTIONS}
        for s in SECTIONS:
            flat.update({f"{s}.{k}": v for k, v in m.get(s, {}).items()})
        grid = {k: v for k, v in flat.items() if isinstance(v, list)}
        for combo in itertools.product(*grid.values()):
            p = dict(flat, **dict(zip(grid, combo)))
            variant = label + "".join(f"_{k.split('.')[-1]}{v}" for k, v in zip(grid, combo))
            params = {k: v for k, v in p.items() if "." not in k}
            for s in SECTIONS:
                params[s] = {k[len(s) + 1:]: v for k, v in p.items() if k.startswith(s + ".")}
            for unit in cfg["units"]:
                out.append((variant, label, params, unit))
    return out


def build_policy(spec, device="cpu"):
    """Unbound policy from a `policy:` block {type: <loader>, **loader arguments}."""
    spec = dict(spec)
    return registry.policy_loader(spec.pop("type", "flow"))(device=device, **spec)


def build_constraint(cfg, overrides):
    """cls(bound_policy, **context) factory from the experiment `constraint:` block and a method's overrides."""
    spec = dict(cfg["constraint"], **overrides)
    return functools.partial(registry.constraint(spec.pop("type")), **spec)


def build_sampler(params):
    fn = registry.sampler(params["sampler"])
    return registry.with_post_projection(fn) if params.get("post_project", False) else fn


def score_loglik(velocity_fn, records, cap, seed=0):
    """Mean log p(x | obs) / D over the samples of a cell (at most `cap`, drawn with a fixed seed)."""
    records = [r for r in records if r.get("score", True)]
    if not records:
        return float("nan"), float("nan"), 0
    x = torch.cat([r["x"].reshape(-1, *r["x"].shape[-2:]) for r in records])
    o = torch.cat([r["obs"].reshape(-1, r["obs"].shape[-1]) for r in records])
    if cap is not None and x.shape[0] > cap:
        idx = torch.randperm(x.shape[0], generator=torch.Generator().manual_seed(seed))[:cap]
        x, o = x[idx], o[idx]
    torch.manual_seed(seed)
    params = list(velocity_fn.parameters()) if isinstance(velocity_fn, torch.nn.Module) else []
    dev = params[0].device if params else torch.device("cpu")
    lp = log_likelihood(velocity_fn, x.to(dev), o.to(dev))
    return float(lp.mean()), float(lp.median()), int(lp.numel())


def run_cell(cfg, cell, device="cpu"):
    variant, label, params, unit = cell
    out_dir = os.path.join(RESULTS, cfg["name"], variant)
    path = os.path.join(out_dir, f"{unit}.json")
    if os.path.exists(path):
        print(f"[skip] {path}")
        return
    bench = registry.benchmark(cfg["benchmark"])
    policy = build_policy(cfg["policy"], device)
    make_constraint = build_constraint(cfg, params["constraint"])
    sampler = build_sampler(params)
    kwargs = {k: v for k, v in params.items() if k not in ("sampler", "post_project") + SECTIONS}
    t0 = time.time()
    records = bench.evaluate_unit(unit, sampler, kwargs, policy, make_constraint, ASSETS,
                                  **dict(cfg.get("eval", {}), **params["eval"]))
    wall = time.time() - t0
    metrics = bench.summarize(records)
    metrics["wall_s_per_rollout"] = wall / max(len(records), 1)
    if cfg.get("logl", True) and getattr(policy, "velocity_fn", None) is not None:
        cap = cfg.get("logl_cap", 400)
        metrics["logl_mean"], metrics["logl_median"], metrics["logl_n"] = score_loglik(policy.velocity_fn, records, cap)
    os.makedirs(out_dir, exist_ok=True)
    torch.save(records, path[:-5] + ".pt")
    with open(path, "w") as f:
        json.dump(dict(config=cfg["name"], variant=variant, method=label, unit=unit, params=params,
                       metrics=metrics), f, indent=2)
    print(f"[done] {path}: " + ", ".join(f"{k}={v:.4g}" for k, v in metrics.items() if isinstance(v, float)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--cell", type=int, default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    cfg = load_config(a.config)
    cells = expand(cfg)
    if a.list:
        print(len(cells))
        return
    for i, c in enumerate(cells):
        if a.cell is None or a.cell == i:
            run_cell(cfg, c, a.device)


if __name__ == "__main__":
    main()
