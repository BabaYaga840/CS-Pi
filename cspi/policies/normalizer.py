"""Per-dimension affine map between physical space and a policy's model space."""
import torch
import torch.nn as nn


class Normalizer(nn.Module):
    """kind="meanstd": x_n = (x - mean) / std                       (2D benchmark)
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
    def identity(cls):
        """Model space = physical space (for policies that do their own normalization)."""
        return cls("meanstd", 0.0, 1.0, 0.0, 1.0)

    @classmethod
    def fit(cls, kind, x, obs):
        """Fit from training tensors x (N, ..., d) and obs (N, obs_dim)."""
        xf = x.reshape(-1, x.shape[-1])
        if kind == "meanstd":
            return cls(kind, xf.mean(0), xf.std(0) + 1e-6, obs.mean(0), obs.std(0) + 1e-6)
        lo, hi = xf.min(0).values, xf.max(0).values
        return cls(kind, lo, lo + (hi - lo).clamp(min=1e-3), obs.min(0).values, obs.max(0).values)
