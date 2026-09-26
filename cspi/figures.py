"""Analyses computed from saved rollouts (no new sampling): run after eval.sh.

    python -m cspi.figures ndynamics     # tab:nproj / Fig. n-dynamics: rollback depth n and correction size per iteration
    python -m cspi.figures fallback      # tab:fb: how often CS-Pi's terminal projection fires, and logL of those samples
    python -m cspi.figures smoothness    # Fig. jerk: acceleration / jerk of executed trajectories vs the prior
    python -m cspi.figures success_logl  # Shelf logL of successful rollouts only, relative to the prior
    python -m cspi.figures all
"""
import argparse
import glob
import os

import numpy as np
import torch

from cspi.flow import load_prior
from cspi.loglik import log_likelihood
from cspi.run import ASSETS, RESULTS

CSPI_CELLS = {  # benchmark label -> (result glob, checkpoint, k used by the cell)
    "2D": ("tab_d3il/CS-Pi/*.pt", "avoid2d", 1.0),
    "2D soft+hard": ("fig_softhard_2d/CS-Pi_k100.0/*.pt", "avoid2d", 100.0),
    "Spheres ID": ("tab_spheres/CS-Pi/id.pt", "spheres", 100.0),
    "Spheres OOD": ("tab_spheres/CS-Pi/ood.pt", "spheres", 100.0),
    "Spheres ID k=10": ("hp_spheres/CS-Pi_k10.0/id.pt", "spheres", 10.0),
    "Orient": ("tab_orient/CS-Pi + EE-prox/*.pt", "orient", 10.0),
    "Shelf": ("tab_shelf/CS-Pi/*.pt", "shelf", 10.0),
}


def _records(pattern):
    out = []
    for f in sorted(glob.glob(os.path.join(RESULTS, pattern))):
        out += torch.load(f)
    return out


def _loops(records):
    """All CS-Pi loops (one per sampler call) as lists of n per iteration."""
    loops = []
    for r in records:
        h = r.get("n_history", [])
        loops += [x for x in h if isinstance(x, list)] if h and isinstance(h[0], list) else [h]
    return [lp for lp in loops if lp]


def ndynamics():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    print("| benchmark | loops | n per iteration (mean over loops still running) | correction |d| first -> last |")
    print("|---|---|---|---|")
    for name, (pattern, _, k) in CSPI_CELLS.items():
        loops = _loops(_records(pattern))
        if not loops:
            continue
        L = max(map(len, loops))
        n = [float(np.mean([lp[i] for lp in loops if len(lp) > i])) for i in range(L)]
        d = [k * (1 - v) / v for v in n]  # n = k / (k + d)
        print(f"| {name} | {len(loops)} | " + ", ".join(f"{v:.3f}" for v in n[:6]) + f" | {d[0]:.3g} -> {d[-1]:.3g} |")
        ax[0].plot(range(1, L + 1), n, marker="o", label=name)
        ax[1].semilogy(range(1, L + 1), d, marker="o", label=name)
    ax[0].set(xlabel="iteration", ylabel="rollback depth n")
    ax[1].set(xlabel="iteration", ylabel="correction size")
    ax[0].legend(fontsize=7)
    os.makedirs(os.path.join(RESULTS, "figures"), exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "figures", "ndynamics.pdf"))


def _flat(records, keep=lambda r, i: True):
    xs, os_ = [], []
    for r in records:
        x, o = r["x"], r["obs"]
        x = x.reshape(-1, *x.shape[-2:])
        o = o.reshape(-1, o.shape[-1])
        flags = r.get("fallback_flags", [r.get("fallback", False)])
        per = x.shape[0] // max(len(flags), 1)  # samples per sampler call
        sel = [i for i in range(x.shape[0]) if keep(r, flags[min(i // per, len(flags) - 1)])]
        xs.append(x[sel])
        os_.append(o[sel])
    return torch.cat(xs), torch.cat(os_)


def _median_logl(model, x, o, cap=400):
    if x.shape[0] == 0:
        return float("nan")
    if x.shape[0] > cap:
        idx = torch.randperm(x.shape[0], generator=torch.Generator().manual_seed(0))[:cap]
        x, o = x[idx], o[idx]
    torch.manual_seed(0)
    return float(log_likelihood(model, x, o).median())


def fallback():
    rows = {"2D": "tab_d3il/{}/*.pt", "Spheres ID": "tab_spheres/{}/id.pt", "Spheres OOD": "tab_spheres/{}/ood.pt",
            "Shelf": "tab_shelf/{}/*.pt"}
    print("| benchmark | fire rate | converged | fallback | projection | prior |  (median logL)")
    print("|---|---|---|---|---|---|")
    for name, pat in rows.items():
        ck = CSPI_CELLS[name if name in CSPI_CELLS else "2D"][1]
        model = load_prior(os.path.join(ASSETS, "checkpoints", f"{ck}.pt"))[0]
        cs = [r for r in _records(pat.format("CS-Pi")) if r.get("score", True)]
        if not cs:
            continue
        flags = [f for r in cs for f in r.get("fallback_flags", [r.get("fallback", False)])]
        conv = _median_logl(model, *_flat(cs, lambda r, f: not f))
        fb = _median_logl(model, *_flat(cs, lambda r, f: f))
        proj = _median_logl(model, *_flat([r for r in _records(pat.format("Projection")) if r.get("score", True)]))
        prior = _median_logl(model, *_flat([r for r in _records(pat.format("Unguided")) if r.get("score", True)]))
        print(f"| {name} | {100 * np.mean(flags):.1f}% | {conv:.2f} | {fb:.2f} | {proj:.2f} | {prior:.2f} |")


def _smooth_2d(records):
    acc, jerk = [], []
    for r in records:
        p = r["traj"]
        a = np.diff(p, n=2, axis=0)
        acc.append(np.linalg.norm(a, axis=-1).max())
        jerk.append(np.linalg.norm(np.diff(p, n=3, axis=0), axis=-1).max())
    return np.array(acc), np.array(jerk)


def _smooth_joint(records):
    acc, jerk = [], []
    for r in records:
        q = r["q_traj"].numpy()
        q = q[None] if q.ndim == 2 else q
        acc.append(np.linalg.norm(np.diff(q, n=2, axis=1), axis=-1).max(1).mean())
        jerk.append(np.linalg.norm(np.diff(q, n=3, axis=1), axis=-1).max(1).mean())
    return np.array(acc), np.array(jerk)


def smoothness():
    print("| benchmark | method | mean max accel | p95 max accel | mean max jerk | p95 max jerk |")
    print("|---|---|---|---|---|---|")
    for bench, pat, fn, methods in [
            ("2D", "tab_d3il/{}/*.pt", _smooth_2d, ["Unguided", "CS-Pi", "Projection", "Guidance", "Trust", "Rejection"]),
            ("Shelf", "tab_shelf/{}/*.pt", _smooth_joint, ["Unguided", "CS-Pi", "Projection", "Guidance", "Trust",
                                                         "Rejection"])]:
        for m in methods:
            recs = _records(pat.format(m))
            if recs:
                a, j = fn(recs)
                print(f"| {bench} | {m} | {a.mean():.4f} | {np.percentile(a, 95):.4f} | {j.mean():.4f} | "
                      f"{np.percentile(j, 95):.4f} |")


def success_logl():
    model = load_prior(os.path.join(ASSETS, "checkpoints", "shelf.pt"))[0]
    base = {}
    print("| method | logL of successful rollouts - prior logL (per scenario, then averaged) |")
    print("|---|---|")
    scen = sorted(os.path.basename(f)[:-3] for f in glob.glob(os.path.join(RESULTS, "tab_shelf/Unguided/*.pt")))
    for s in scen:
        x, o = _flat(torch.load(os.path.join(RESULTS, f"tab_shelf/Unguided/{s}.pt")))
        torch.manual_seed(0)
        base[s] = float(log_likelihood(model, x[:400], o[:400]).mean())
    for m in ["CS-Pi", "Rejection", "Projection", "Guidance", "Trust"]:
        deltas = []
        for s in scen:
            f = os.path.join(RESULTS, f"tab_shelf/{m}/{s}.pt")
            succ = [r for r in torch.load(f) if r["safe"]] if os.path.exists(f) else []
            if succ:
                x, o = _flat(succ)
                torch.manual_seed(0)
                deltas.append(float(log_likelihood(model, x[:400], o[:400]).mean()) - base[s])
        if deltas:
            print(f"| {m} | {np.mean(deltas):+.2f} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["ndynamics", "fallback", "smoothness", "success_logl", "all"])
    a = ap.parse_args()
    for name in (["ndynamics", "fallback", "smoothness", "success_logl"] if a.what == "all" else [a.what]):
        print(f"\n## {name}")
        globals()[name]()


if __name__ == "__main__":
    main()
