"""Velocity-field networks v_theta(x_t, t, obs) used by the paper's priors."""
import math

import torch
import torch.nn as nn


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


def build_model(cfg):
    """Instantiate the network described by a checkpoint's `config` dict."""
    if cfg["arch"] == "mlp":
        return FlowMLP(cfg["action_dim"], cfg["obs_dim"], cfg["horizon"], cfg.get("hidden_dim", 512),
                       cfg.get("n_layers", 4), cfg.get("time_embed_dim", 64))
    return FlowUNet(cfg["action_dim"], cfg["obs_dim"], cfg["horizon"], cfg.get("time_embed_dim", 64),
                    tuple(cfg.get("down_dims", (64, 128, 256))), cond_mode=cfg.get("cond_mode", "concat"),
                    obs_embed_dim=cfg.get("obs_embed_dim", 128))
