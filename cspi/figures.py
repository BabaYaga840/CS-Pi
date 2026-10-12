"""Analyses computed from saved rollouts (no new sampling): run after eval.sh.

    python -m cspi.figures ndynamics     # tab:nproj / Fig. n-dynamics: rollback depth n and correction size per iteration
    python -m cspi.figures fallback      # tab:fb: how often CS-Pi's terminal projection fires, and logL of those samples
    python -m cspi.figures smoothness    # Fig. jerk: acceleration / jerk of executed trajectories vs the prior
    python -m cspi.figures success_logl  # Shelf logL of successful rollouts only, relative to the prior
    python -m cspi.figures onoff         # terminal projection on vs off: Safe-SR, logL, fire rate
    python -m cspi.figures fixedk        # one k (= 20) on every benchmark vs the tuned k
    python -m cspi.figures softhard      # soft + hard composition: both met, collision rate, |y - y_ref|
    python -m cspi.figures earlystop     # stopping rules: Safe-SR vs NFE per cell (CSV for the Pareto plot)
    python -m cspi.figures walltime      # raw per-rollout wall-clock (CSV)
    python -m cspi.figures all
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

from cspi.flow import load_prior
from cspi.loglik import log_likelihood
from cspi.run import ASSETS, RESULTS

CSPI_CELLS = {  # benchmark label -> (result glob, checkpoint)
    "2D": ("tab_d3il/CS-Pi/*.pt", "avoid2d"),
    "2D soft+hard": ("fig_softhard_2d/CS-Pi_k100.0/*.pt", "avoid2d"),
    "Spheres ID": ("tab_spheres/CS-Pi/id.pt", "spheres"),
    "Spheres OOD": ("tab_spheres/CS-Pi/ood.pt", "spheres"),
    "Spheres ID k=10": ("hp_spheres/CS-Pi_k10.0/id.pt", "spheres"),
    "Orient": ("tab_orient/CS-Pi + EE-prox/*.pt", "orient"),
    "Orient k=1": ("fixedk_orient/CS-Pi + EE-prox_k1.0/*.pt", "orient"),
    "Shelf": ("tab_shelf/CS-Pi/*.pt", "shelf"),
    "Shelf k=1": ("fixedk_shelf/CS-Pi_k1.0/*.pt", "shelf"),
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
    logs = {}
    for name, (pattern, _) in CSPI_CELLS.items():
        loops = _loops(_records(pattern))
        if not loops:
            continue
        logs[name] = loops
        k = json.load(open(glob.glob(os.path.join(RESULTS, pattern))[0][:-3] + ".json"))["params"]["k"]
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
    with open(os.path.join(RESULTS, "figures", "ndynamics_loops.json"), "w") as f:
        json.dump(logs, f)  # benchmark -> list of loops, each the rollback depth n per iteration


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
        ck = CSPI_CELLS[name][1]
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


def _smooth_spline(records, split):
    """Sphere-Avoid: decode the control points to the dense 64-step trajectory that is evaluated."""
    from cspi.benchmarks.spheres import load
    task, _ = load(split)
    norm = load_prior(os.path.join(ASSETS, "checkpoints", "spheres.pt"))[1]
    tasks = torch.load(os.path.join(ASSETS, "tasks", "spheres.pt"))
    acc, jerk = [], []
    for r, q_s, q_g in zip(records, tasks["q_start"], tasks["q_goal"]):
        if not r["endpoints_feasible"]:
            continue
        ctrl = norm.unnormalize(r["x"])
        q = task.parametric_trajectory.get_q_trajectory(ctrl, q_s, q_g, get_type=("pos",))["pos"].detach().numpy()
        acc.append(np.linalg.norm(np.diff(q, n=2, axis=1), axis=-1).max(1).mean())
        jerk.append(np.linalg.norm(np.diff(q, n=3, axis=1), axis=-1).max(1).mean())
    return np.array(acc), np.array(jerk)


def smoothness():
    print("| benchmark | method | mean max accel | p95 max accel | mean max jerk | p95 max jerk |")
    print("|---|---|---|---|---|---|")
    for bench, pat, fn, methods in [
            ("2D", "tab_d3il/{}/*.pt", _smooth_2d, ["Unguided", "CS-Pi", "Projection", "Guidance", "Trust", "Rejection"]),
            ("Shelf", "tab_shelf/{}/*.pt", _smooth_joint, ["Unguided", "CS-Pi", "Projection", "Guidance", "Trust",
                                                         "Rejection"]),
            ("Spheres ID", "tab_spheres/{}/id.pt", lambda r: _smooth_spline(r, "id"),
             ["Unguided", "CS-Pi", "Projection", "Guidance", "Trust", "Rejection"]),
            ("Spheres OOD", "tab_spheres/{}/ood.pt", lambda r: _smooth_spline(r, "ood"),
             ["Unguided", "CS-Pi", "Projection", "Guidance", "Trust", "Rejection"]),
            ("Orient", "tab_orient/{}/*.pt", _smooth_joint,
             ["Unguided", "CS-Pi + EE-prox", "Guidance a=0.003", "Guidance a=0.03", "Trust"])]:
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


def _cell(path):
    return json.load(open(path))["metrics"] if os.path.exists(path) else None


def _macro(pattern, keys):
    ms = [json.load(open(f))["metrics"] for f in sorted(glob.glob(os.path.join(RESULTS, pattern)))]
    return {k: float(np.mean([m[k] for m in ms])) if ms and all(k in m for m in ms) else float("nan") for k in keys}


ONOFF = {"2D": ("tab_d3il/CS-Pi/*.json", "tpoff_d3il/CS-Pi (no terminal proj.)/*.json"),
         "Spheres L1 (ID)": ("tab_spheres/CS-Pi/id.json", "tpoff_spheres/CS-Pi (no terminal proj.)/id.json"),
         "Spheres L2 (OOD)": ("tab_spheres/CS-Pi/ood.json", "tpoff_spheres/CS-Pi (no terminal proj.)/ood.json"),
         "Shelf": ("tab_shelf/CS-Pi/*.json", "tpoff_shelf/CS-Pi (no terminal proj.)/*.json")}


def onoff():
    keys = ["success", "logl_mean", "logl_median", "fallback_rate", "nfe"]
    print("| benchmark | terminal proj. | Safe-SR | logL mean | logL median | fire rate | NFE |")
    print("|---|---|---|---|---|---|---|")
    for name, (on, off) in ONOFF.items():
        for tag, pat in (("on", on), ("off", off)):
            m = _macro(pat, keys)
            fire = f"{100 * m['fallback_rate']:.1f}%" if tag == "on" else "— (disabled)"
            print(f"| {name} | {tag} | {100 * m['success']:.1f} | {m['logl_mean']:.2f} | {m['logl_median']:.2f} | "
                  f"{fire} | {m['nfe']:.0f} |")


FIXEDK = {"2D": ("tab_d3il/CS-Pi/*.json", "tab_d3il/CS-Pi/*.json"),
          "Spheres ID": ("tab_spheres/CS-Pi/id.json", "fixedk_spheres/CS-Pi/id.json"),
          "Spheres OOD": ("tab_spheres/CS-Pi/ood.json", "fixedk_spheres/CS-Pi/ood.json"),
          "Shelf": ("tab_shelf/CS-Pi/*.json", "fixedk_shelf/CS-Pi_k20.0/*.json"),
          "Orient": ("tab_orient/CS-Pi + EE-prox/*.json", "fixedk_orient/CS-Pi + EE-prox_k20.0/*.json")}


def fixedk():
    print("| benchmark | tuned k | SR (tuned) | logL (tuned) | SR (k=20) | logL (k=20) | SR loss (pp) |")
    print("|---|---|---|---|---|---|---|")
    for name, (tuned, fixed) in FIXEDK.items():
        f = sorted(glob.glob(os.path.join(RESULTS, tuned)))
        if not f:
            continue
        k = json.load(open(f[0]))["params"]["k"]
        a, b = _macro(tuned, ["success", "logl_mean"]), _macro(fixed, ["success", "logl_mean"])
        print(f"| {name} | {k:g} | {100 * a['success']:.1f} | {a['logl_mean']:.2f} | {100 * b['success']:.1f} | "
              f"{b['logl_mean']:.2f} | {100 * (a['success'] - b['success']):+.1f} |")


def softhard():
    from cspi.benchmarks.avoid2d import START, waypoints
    norm = load_prior(os.path.join(ASSETS, "checkpoints", "avoid2d.pt"))[1]
    y_ref = 0.7
    print("| method | both met | collision rate | mean abs y-error | median abs y-error | NFE |")
    print("|---|---|---|---|---|---|")
    for d in sorted(glob.glob(os.path.join(RESULTS, "fig_softhard_2d", "*"))):
        recs = _records(os.path.join("fig_softhard_2d", os.path.basename(d), "*.pt"))
        if not recs:
            continue
        x = torch.cat([r["x"] for r in recs])
        wp = waypoints(norm.unnormalize(x), torch.tensor([START]).expand(x.shape[0], 2))
        yerr = (wp[..., 1] - y_ref).abs().mean(-1).numpy()
        both = np.mean([r["success"] for r in recs])
        coll = 1.0 - np.mean([r["collision_free"] for r in recs])
        nfe = np.mean([r["nfe"] for r in recs])
        print(f"| {os.path.basename(d)} | {100 * both:.1f} | {100 * coll:.1f} | {yerr.mean():.4f} | "
              f"{np.median(yerr):.4f} | {nfe:.0f} |")


def earlystop():
    out = os.path.join(RESULTS, "figures", "earlystop_scatter.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    n = 0
    with open(out, "w") as f:
        f.write("config,rule,k,unit,success,nfe\n")
        for cfg in ("earlystop_d3il", "earlystop_spheres", "earlystop_shelf"):
            for path in sorted(glob.glob(os.path.join(RESULTS, cfg, "*", "*.json"))):
                d = json.load(open(path))
                f.write(f"{cfg},{d['method']},{d['params']['k']},{d['unit']},{d['metrics']['success']},"
                        f"{d['metrics']['nfe']}\n")
                n += 1
    print(f"wrote {n} cells to {out}")


def walltime():
    out = os.path.join(RESULTS, "figures", "walltime_per_rollout.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    n = 0
    with open(out, "w") as f:
        f.write("config,method,unit,rollout,wall_s,nfe\n")
        for path in sorted(glob.glob(os.path.join(RESULTS, "walltime_*", "*", "*.pt"))):
            cfg, method = path.split(os.sep)[-3:-1]
            unit = os.path.basename(path)[:-3]
            for i, r in enumerate(torch.load(path)):
                if "wall_s" in r:
                    f.write(f"{cfg},{method},{unit},{i},{r['wall_s']:.6f},{r['nfe']}\n")
                    n += 1
    print(f"wrote {n} rollouts to {out}")


def main():
    ap = argparse.ArgumentParser()
    names = ["ndynamics", "fallback", "smoothness", "success_logl", "onoff", "fixedk", "softhard", "earlystop",
             "walltime"]
    ap.add_argument("what", choices=names + ["all"])
    a = ap.parse_args()
    for name in (names if a.what == "all" else [a.what]):
        print(f"\n## {name}")
        globals()[name]()


if __name__ == "__main__":
    main()
