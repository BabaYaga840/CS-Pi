"""The policy contract: what CS-Pi and the baselines need from a flow or diffusion policy.

Time convention used everywhere: t = 0 is the noise (source) distribution, t = 1 is data.

A `Policy` is first loaded *unbound* (network + normalizer). Before sampling, a benchmark binds it
to one batch of observations with `condition(obs)`; samplers only ever see bound policies. Samples
live in the policy's *model space* (usually normalized); `normalizer` maps them to physical units.

Required by CS-Pi, Projection, Rejection and Unguided (a "black-box" policy is enough):

    condition(obs)             -> bound policy for raw observations obs (B, obs_dim)
    shape                      (B, H, d) shape of one batch of samples          [bound]
    obs_n                      (B, obs_dim) conditioning in model space        [bound]
    nfe                        network evaluations so far; samplers report differences
    normalizer                 Normalizer between model space and physical units
    noise()                    -> a draw from the source distribution, shape `shape`
    sample(step_size)          -> a full sample, t: 0 -> 1
    denoise(x, t0, step_size)  -> finish a sample that sits at time t0, t: t0 -> 1
    renoise(x1, t, eps)        -> put a clean sample x1 back at time t (eps=None: no noise)

Additionally required by gradient guidance, Trust Sampling, SafeDiffuser, SafeFlowMatcher:

    velocity(x, t)             -> dx/dt of the sampling ODE at time t (counts one NFE)

and by the logL metric: `velocity_fn(x, t_vec, obs_n)`, the raw network (not counted).

Step-size convention (kept from the paper code): a full sample takes max(1, round(1 / step_size))
steps; finishing from t0 takes max(1, ceil((1 - t0) / step_size)) steps on the same grid.
"""
import math


def full_steps(step_size):
    return max(1, round(1.0 / step_size))


def partial_steps(t0, step_size):
    return max(1, math.ceil((1.0 - t0) / step_size))


class Policy:
    """Base class. Subclasses implement condition / noise / denoise (and velocity if they can)."""

    shape = None
    obs_n = None
    nfe = 0
    normalizer = None

    def condition(self, obs):
        raise NotImplementedError

    def noise(self):
        raise NotImplementedError

    def denoise(self, x, t0, step_size=0.1):
        raise NotImplementedError

    def sample(self, step_size=0.1):
        return self.denoise(self.noise(), 0.0, step_size)

    def renoise(self, x1, t, eps=None):
        """Rectified-flow interpolant x_t = t x1 + (1 - t) eps; override for other noise schedules."""
        return x1 * t + (eps * (1.0 - t) if eps is not None else 0.0)

    def velocity(self, x, t):
        raise NotImplementedError(f"{type(self).__name__} exposes no velocity field; gradient guidance, "
                                  "Trust Sampling and the CBF baselines need one (CS-Pi does not)")
