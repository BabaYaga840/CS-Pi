"""Print a paper table from the cells written by cspi.run.

    python -m cspi.tables configs/tab_shelf.yaml            # markdown
    python -m cspi.tables configs/tab_shelf.yaml --latex    # LaTeX rows
    python -m cspi.tables configs/tab_shelf.yaml --per-unit # one column group per scenario

The config's `table` block chooses the columns ({header: metric}) and how units combine:
`per_unit: false` (default) averages every metric over the units (macro average over
scenarios or seeds); `per_unit: true` prints one column group per unit (Sphere-Avoid ID / OOD).
Metrics listed under `percent` are multiplied by 100.
"""
import argparse
import json
import os

import numpy as np

from cspi.run import RESULTS, expand, load_config


def collect(cfg):
    rows = {}
    for variant, label, params, unit in expand(cfg):
        path = os.path.join(RESULTS, cfg["name"], variant, f"{unit}.json")
        rows.setdefault(variant, {})[str(unit)] = json.load(open(path))["metrics"] if os.path.exists(path) else None
    return rows


def fmt(v, pct):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if pct:
        return f"{100 * v:.1f}"
    return f"{v:.2e}" if abs(v) >= 1e6 else (f"{v:.0f}" if abs(v) >= 100 else f"{v:.2f}")


def build(cfg):
    t = cfg.get("table", {})
    cols, pct = t.get("columns", {"SR": "success", "NFE": "nfe"}), set(t.get("percent", ["success"]))
    units = [str(u) for u in cfg["units"]]
    groups = units if t.get("per_unit", False) else ["all"]
    header = [f"{g} {h}" if len(groups) > 1 else h for g in groups for h in cols]
    body = []
    for variant, per_unit in collect(cfg).items():
        cells = []
        for g in groups:
            use = [per_unit[u] for u in (units if g == "all" else [g])]
            for m in cols.values():
                vals = [r[m] for r in use if r is not None and m in r]
                cells.append(fmt(float(np.mean(vals)) if len(vals) == len(use) else None, m in pct))
        body.append([variant] + cells)
    return ["Method"] + header, body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--per-unit", action="store_true", help="one column group per unit instead of the macro average")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.per_unit:
        cfg.setdefault("table", {})["per_unit"] = True
    head, body = build(cfg)
    if a.latex:
        for row in body:
            print(" & ".join(row) + r" \\")
        return
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for row in body:
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
