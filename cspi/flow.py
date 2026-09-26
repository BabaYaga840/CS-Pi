"""Flow-matching prior: velocity-field networks, normalizer, OT-CFM loss, Euler solver.

Convention used everywhere in this repo (rectified / OT-CFM interpolant):

    x_t = (1 - t) * eps + t * x_1 ,   eps ~ N(0, I),   t in [0, 1]
    target velocity  v = x_1 - eps

so t = 0 is pure noise and t = 1 is data. Samples are produced by integrating
dx/dt = v_theta(x, t, obs) forward from t = 0 with explicit Euler steps.
Every sampler in `cspi.samplers` and `cspi.cbf` works in the *normalized*
model space; benchmarks map to and from physical space with `Normalizer`.
"""
import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freq = torch.exp(torch.arange(half, device=t.device) * -(math.log(10000) / (half - 1)))
        emb = t[:, None] * freq[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


def _time_mlp(dim):
    return nn.Sequential(SinusoidalPosEmb(dim), nn.Linear(dim, dim * 2), nn.SiLU(),
                         nn.Linear(dim * 2, dim))


class FlowMLP(nn.Module):
    """MLP velocity field on a flattened (horizon, action_dim) chunk (2D benchmark)."""

    def __init__(self, action_dim, obs_dim, horizon, hidden_dim=512, n_layers=4, time_embed_dim=64):
        super().__init__()
        self.horizon, self.action_dim = horizon, action_dim
        self.time_embed = _time_mlp(time_embed_dim)
        layers, d_in = [], horizon * action_dim + time_embed_dim + obs_dim
        for i in range(n_layers):
            layers += [nn.Linear(d_in if i == 0 else hidden_dim, hidden_dim), nn.SiLU()]
        layers.append(nn.Linear(hidden_dim, horizon * action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x_t, t, obs):
        B = x_t.shape[0]
        h = torch.cat([x_t.reshape(B, -1), self.time_embed(t), obs], dim=-1)
        return self.net(h).reshape(B, self.horizon, self.action_dim)


class FlowUNet(nn.Module):
    """1D temporal U-Net (no temporal downsampling) with additive conditioning.

    cond = cat(time_embedding(t), obs) or, with cond_mode="obs_mlp",
    cat(time_embedding(t), MLP(obs)). Each block adds a per-channel bias W(cond).
    Used by the three Panda benchmarks.
    """

    def __init__(self, action_dim, obs_dim, horizon, time_embed_dim=64, down_dims=(64, 128, 256),
                 kernel_size=3, cond_mode="concat", obs_embed_dim=128):
        super().__init__()
        assert cond_mode in ("concat", "obs_mlp"), cond_mode
        self.horizon, self.action_dim = horizon, action_dim
        self.time_embed = _time_mlp(time_embed_dim)
        if cond_mode == "obs_mlp":
            self.obs_encoder = nn.Sequential(nn.Linear(obs_dim, obs_embed_dim), nn.SiLU(),
                                             nn.Linear(obs_embed_dim, obs_embed_dim), nn.SiLU())
            cond_dim = time_embed_dim + obs_embed_dim
        else:
            self.obs_encoder = None
            cond_dim = time_embed_dim + obs_dim
        pad = kernel_size // 2

        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv1d(c_in, c_out, kernel_size, padding=pad), nn.GroupNorm(min(8, c_out), c_out), nn.SiLU(),
                nn.Conv1d(c_out, c_out, kernel_size, padding=pad), nn.GroupNorm(min(8, c_out), c_out), nn.SiLU())

        self.input_proj = nn.Conv1d(action_dim, down_dims[0], 1)
        self.encoders = nn.ModuleList(block(down_dims[i], down_dims[i + 1]) for i in range(len(down_dims) - 1))
        self.enc_cond_proj = nn.ModuleList(nn.Linear(cond_dim, c) for c in down_dims[1:])
        ups = range(len(down_dims) - 1, 0, -1)
        self.decoders = nn.ModuleList(block(down_dims[i] + down_dims[i - 1], down_dims[i - 1]) for i in ups)
        self.dec_cond_proj = nn.ModuleList(nn.Linear(cond_dim, down_dims[i - 1]) for i in ups)
        self.output_proj = nn.Conv1d(down_dims[0], action_dim, 1)

    def forward(self, x_t, t, obs):
        if self.obs_encoder is not None:
            obs = self.obs_encoder(obs)
        cond = torch.cat([self.time_embed(t), obs], dim=-1)
        h = self.input_proj(x_t.permute(0, 2, 1))
        skips = [h]
        for enc, proj in zip(self.encoders, self.enc_cond_proj):
            h = enc(h) + proj(cond).unsqueeze(-1)
            skips.append(h)
        skips.pop()
        for dec, proj in zip(self.decoders, self.dec_cond_proj):
            h = dec(torch.cat([h, skips.pop()], dim=1)) + proj(cond).unsqueeze(-1)
        return self.output_proj(h).permute(0, 2, 1)


# ---------------------------------------------------------------------------
# Normalizer
# ---------------------------------------------------------------------------

class Normalizer(nn.Module):
    """Per-dimension affine map between physical space and model space.

    kind="meanstd": x_n = (x - mean) / std                       (2D benchmark)
    kind="minmax" : x_n = 2 (x - lo) / (hi - lo) - 1  in [-1, 1]  (Panda benchmarks)
    Stats: `x_a, x_b` for the sample (per last dim), `o_a, o_b` for the conditioning,
    where (a, b) = (mean, std) or (lo, hi).
    """

    def __init__(self, kind, x_a, x_b, o_a, o_b):
        super().__init__()
        assert kind in ("meanstd", "minmax")
        self.kind = kind
        for name, v in (("x_a", x_a), ("x_b", x_b), ("o_a", o_a), ("o_b", o_b)):
            self.register_buffer(name, torch.as_tensor(v, dtype=torch.float32).clone())

    def _fwd(self, x, a, b):
        if self.kind == "meanstd":
            return (x - a.to(x.device)) / b.to(x.device)
        return 2.0 * (x - a.to(x.device)) / (b - a).clamp(min=1e-6).to(x.device) - 1.0

    def _inv(self, x, a, b):
        if self.kind == "meanstd":
            return x * b.to(x.device) + a.to(x.device)
        return (x + 1.0) / 2.0 * (b - a).clamp(min=1e-6).to(x.device) + a.to(x.device)

    def normalize(self, x):
        return self._fwd(x, self.x_a, self.x_b)

    def unnormalize(self, x):
        return self._inv(x, self.x_a, self.x_b)

    def normalize_obs(self, obs):
        return self._fwd(obs, self.o_a, self.o_b)

    @classmethod
    def fit(cls, kind, x, obs):
        """Fit from training tensors x (N, ..., d) and obs (N, obs_dim)."""
        xf = x.reshape(-1, x.shape[-1])
        if kind == "meanstd":
            return cls(kind, xf.mean(0), xf.std(0) + 1e-6, obs.mean(0), obs.std(0) + 1e-6)
        lo, hi = xf.min(0).values, xf.max(0).values
        return cls(kind, lo, lo + (hi - lo).clamp(min=1e-3), obs.min(0).values, obs.max(0).values)


# ---------------------------------------------------------------------------
# Training objective and sampling
# ---------------------------------------------------------------------------

def cfm_loss(model, x1, obs, weight=None):
    """OT-CFM loss on normalized data x1 (B, H, d) and normalized obs (B, obs_dim).

    `weight` (B, H) optionally re-weights timesteps of the chunk (Shelf grasp weighting).
    """
    t = torch.rand(x1.shape[0], device=x1.device)
    eps = torch.randn_like(x1)
    x_t = (1 - t[:, None, None]) * eps + t[:, None, None] * x1
    err = (model(x_t, t, obs) - (x1 - eps)) ** 2
    if weight is None:
        return err.mean()
    w = weight.unsqueeze(-1)
    return (w * err).sum() / (w.sum() * err.shape[-1])


def ode_solve(velocity, x, t0=0.0, t1=1.0, n_steps=10):
    """Explicit Euler for dx/dt = velocity(x, t) from t0 to t1."""
    dt = (t1 - t0) / n_steps
    t = t0
    for _ in range(n_steps):
        x = x + velocity(x, t) * dt
        t += dt
    return x


def build_model(cfg):
    """Instantiate the network described by a checkpoint's `config` dict."""
    if cfg["arch"] == "mlp":
        return FlowMLP(cfg["action_dim"], cfg["obs_dim"], cfg["horizon"], cfg.get("hidden_dim", 512),
                       cfg.get("n_layers", 4), cfg.get("time_embed_dim", 64))
    return FlowUNet(cfg["action_dim"], cfg["obs_dim"], cfg["horizon"], cfg.get("time_embed_dim", 64),
                    tuple(cfg.get("down_dims", (64, 128, 256))), cond_mode=cfg.get("cond_mode", "concat"),
                    obs_embed_dim=cfg.get("obs_embed_dim", 128))


def load_prior(path, device="cpu"):
    """Load a released checkpoint -> (model in eval mode, Normalizer, config dict)."""
    ck = torch.load(path, map_location=device)
    model = build_model(ck["config"])
    model.load_state_dict(ck["model"])
    norm = Normalizer(**ck["normalizer"])
    return model.to(device).eval(), norm.to(device), ck["config"]


def save_prior(path, model, normalizer, config, **extra):
    torch.save({"config": config, "model": model.state_dict(),
                "normalizer": {"kind": normalizer.kind, "x_a": normalizer.x_a.cpu(), "x_b": normalizer.x_b.cpu(),
                               "o_a": normalizer.o_a.cpu(), "o_b": normalizer.o_b.cpu()}, **extra}, path)
