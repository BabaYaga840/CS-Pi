"""Train a flow-matching prior (OT-CFM) from a released training file.

    python -m cspi.train configs/train/avoid2d.yaml [--device cuda]

Training files (assets/data/*_train.pt) come in two layouts:
  samples : x (N, H, d) and obs (N, obs_dim)                       (2D chunks, Sphere-Avoid control points)
  windows : dense (N, T, d), context (N, c), optional weight (N, T) (Orient, Shelf joint trajectories)
            every epoch draws `windows_per_traj` random windows q[t:t+H] per trajectory,
            conditioned on obs = (q[t], context).
10% of the trajectories are held out; the EMA model with the lowest held-out loss is saved.
`micro_batch` (optional) splits each batch for gradient accumulation on smaller GPUs.
"""
import argparse
import copy
import os

import torch
import yaml

from cspi.flow import Normalizer, build_model, cfm_loss, save_prior


def windows(dense, context, weight, horizon, repeats):
    N, T, _ = dense.shape
    idx = torch.arange(N).repeat(repeats)
    t0 = torch.randint(0, T - horizon + 1, (len(idx),))
    steps = t0[:, None] + torch.arange(horizon)
    x = dense[idx[:, None], steps]
    obs = torch.cat([dense[idx, t0], context[idx]], -1)
    w = weight[idx[:, None], steps] if weight is not None else None
    return x, obs, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    torch.manual_seed(cfg.get("seed", 42))
    data = torch.load(cfg["data"])
    n = (data["x"] if "x" in data else data["dense"]).shape[0]
    perm = torch.randperm(n)
    val, tr = perm[: n // 10], perm[n // 10:]
    split = {k: (v[tr], v[val]) for k, v in data.items() if torch.is_tensor(v) and v.shape[0] == n}

    def batches(part, repeats=1):
        s = {k: v[part] for k, v in split.items()}
        if "x" in s:
            return s["x"], s["obs"], None
        return windows(s["dense"], s["context"], s.get("weight"), cfg["model"]["horizon"], repeats)

    x_all, o_all, _ = batches(0, cfg.get("windows_per_traj", 1))
    norm = Normalizer.fit(cfg["normalizer"], x_all, o_all).to(a.device)
    model = build_model(cfg["model"]).to(a.device)
    ema = copy.deepcopy(model).eval()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    best = float("inf")
    os.makedirs(os.path.dirname(cfg["out"]), exist_ok=True)
    for epoch in range(cfg["epochs"]):
        model.train()
        x, o, w = batches(0, cfg.get("windows_per_traj", 1))
        order = torch.randperm(x.shape[0])
        mb = cfg.get("micro_batch", cfg["batch_size"])  # gradient accumulation for large batches
        for i in range(0, len(order) - cfg["batch_size"] + 1, cfg["batch_size"]):
            opt.zero_grad()
            for j in range(i, i + cfg["batch_size"], mb):
                b = order[j:min(j + mb, i + cfg["batch_size"])]
                loss = cfm_loss(model, norm.normalize(x[b].to(a.device)), norm.normalize_obs(o[b].to(a.device)),
                                None if w is None else w[b].to(a.device))
                (loss * len(b) / cfg["batch_size"]).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                for pe, p in zip(ema.parameters(), model.parameters()):
                    pe.mul_(cfg.get("ema", 0.999)).add_(p, alpha=1 - cfg.get("ema", 0.999))
        sched.step()
        with torch.no_grad():
            xv, ov, wv = batches(1, cfg.get("windows_per_traj", 1))
            vl = sum(cfm_loss(ema, norm.normalize(xv[j:j + mb].to(a.device)), norm.normalize_obs(ov[j:j + mb].to(a.device)),
                              None if wv is None else wv[j:j + mb].to(a.device)).item() * len(xv[j:j + mb])
                     for j in range(0, len(xv), mb)) / len(xv)
        if vl < best:
            best = vl
            save_prior(cfg["out"], ema, norm, cfg["model"], epoch=epoch + 1)
        if epoch % 10 == 0 or epoch == cfg["epochs"] - 1:
            print(f"epoch {epoch + 1}: train {loss.item():.4f}  val {vl:.4f}  best {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
