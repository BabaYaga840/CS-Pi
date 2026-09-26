"""Run the experiments behind one paper table or figure.

    python -m cspi.run configs/tab_shelf.yaml            # all cells in sequence (finished cells are skipped)
    python -m cspi.run configs/tab_shelf.yaml --list     # print the number of cells (size of a SLURM array)
    python -m cspi.run configs/tab_shelf.yaml --cell 7   # one cell

A config lists `methods` (label -> sampler and hyperparameters) and `units` (the scenarios,
splits or seeds a benchmark is evaluated on). A list-valued hyperparameter becomes a sweep.
Every (method variant, unit) pair is one cell; it writes results/<name>/<variant>/<unit>.json
(metrics, including the prior's log-likelihood of the samples) and a .pt file with the rollouts.
"""
import argparse
import importlib
import itertools
import json
import os
import time

import torch
import yaml

from cspi import cbf, samplers
from cspi.flow import load_prior
from cspi.loglik import log_likelihood

ASSETS = os.environ.get("CSPI_ASSETS", "assets")
RESULTS = os.environ.get("CSPI_RESULTS", "results")
SAMPLERS = dict(samplers.SAMPLERS, **cbf.SAMPLERS)


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
    """All cells of a config: (variant, method label, params, unit)."""
    out = []
    for label, m in cfg["methods"].items():
        flat = dict(m, **{f"problem.{k}": v for k, v in m.get("problem", {}).items()})
        flat.pop("problem", None)
        grid = {k: v for k, v in flat.items() if isinstance(v, list)}
        for combo in itertools.product(*grid.values()):
            p = dict(flat, **dict(zip(grid, combo)))
            variant = label + "".join(f"_{k.split('.')[-1]}{v}" for k, v in zip(grid, combo))
            params = {k: v for k, v in p.items() if not k.startswith("problem.")}
            params["problem"] = {k[8:]: v for k, v in p.items() if k.startswith("problem.")}
            for unit in cfg["units"]:
                out.append((variant, label, params, unit))
    return out


def score_loglik(model, records, cap, seed=0):
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
    dev = next(model.parameters()).device
    lp = log_likelihood(model, x.to(dev), o.to(dev))
    return float(lp.mean()), float(lp.median()), int(lp.numel())


def run_cell(cfg, cell, device="cpu"):
    variant, label, params, unit = cell
    out_dir = os.path.join(RESULTS, cfg["name"], variant)
    path = os.path.join(out_dir, f"{unit}.json")
    if os.path.exists(path):
        print(f"[skip] {path}")
        return
    bench = importlib.import_module(f"cspi.benchmarks.{cfg['benchmark']}")
    model, norm, _ = load_prior(os.path.join(ASSETS, "checkpoints", f"{cfg['checkpoint']}.pt"), device)
    sampler = SAMPLERS[params["sampler"]]
    kwargs = {k: v for k, v in params.items() if k not in ("sampler", "problem")}
    t0 = time.time()
    records = bench.evaluate_unit(unit, sampler, kwargs, model, norm, ASSETS,
                                  **params["problem"], **cfg.get("eval", {}))
    wall = time.time() - t0
    metrics = bench.summarize(records)
    metrics["wall_s_per_rollout"] = wall / max(len(records), 1)
    if cfg.get("logl", True):
        cap = cfg.get("logl_cap", 400)
        metrics["logl_mean"], metrics["logl_median"], metrics["logl_n"] = score_loglik(model, records, cap)
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
