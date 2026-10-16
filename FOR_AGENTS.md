# FOR_AGENTS.md — using CS-Pi with any flow or diffusion policy

This file is for AI agents (and people) who need to run CS-Pi on a new policy, constraint or task
without reading the whole codebase. Read sections 1–3 before touching code. Section 6 is the
hyperparameter reference.

## 1. Mental model

Everything is one call:

```python
x, info = sampler(policy, constraint, **hyperparameters)   # x in the policy's model space
```

- **`policy`**: a generative policy bound to one batch of observations. It can draw samples,
  put a clean sample back at noise level `t`, and finish a sample from `t`. See
  `cspi/policies/base.py`.
- **`constraint`**: what "feasible" means for this decision, measured by `violation(x)`, plus
  `project(x)`, which moves a sample onto (or towards) the feasible set. See
  `cspi/constraints/base.py`.
- **`sampler`**: CS-Pi (`cspi/core/algorithm.py`) or a baseline (`cspi/baselines/`).
- `info["nfe"]` is the number of network evaluations spent in the call.

CS-Pi (Algorithm 1), written against the contracts:

```
x <- policy.sample()                                   # t: 0 -> 1
while max_b constraint.violation(x) > tau and at most k_max times:
    g     <- constraint.project(x)
    delta <- max_b ||g - x||                            # model-space distance, batch max
    n     <- k / (k + delta)                            # big correction -> deep rollback
    x     <- policy.denoise(policy.renoise(g, n, eps), from t = n)
if still violating and terminal_projection: x <- constraint.fallback(x)
```

A **benchmark** (`cspi/benchmarks/`) owns the environment: tasks, the closed-loop rollout and the
metrics. At every decision it binds the policy to the current observation, builds the constraint
with the current context, calls the sampler and executes the result. An **experiment YAML**
(`configs/`) picks the benchmark, the policy, the constraint and the methods.

## 2. Layout

```
cspi/
  policies/        the policy contract + implementations
    base.py          Policy contract (read this)
    flow.py          FlowPolicy: any velocity field v(x, t, obs); Euler solver; checkpoint I/O; OT-CFM loss
    diffusion.py     DDIMPolicy: adapter for epsilon-prediction diffusion models
    networks.py      FlowMLP, FlowUNet (the paper's priors)
    normalizer.py    Normalizer (model space <-> physical units)
  constraints/     the constraint contract + every constraint / projection used in the paper
    base.py          Constraint contract, ActionSpaceConstraint helper, IdentityConstraint (read this)
    obstacles_2d.py  Obstacles2D, Coupled2D               (2D Avoidance)
    spline_collision.py  SplineCollision                  (Sphere-Avoid)
    orientation.py   OrientationLock                      (3D-Orient)
    chunk_collision.py   ChunkCollision                   (Shelf)
    panda.py         shared Panda tools: MPDCost, prox_lbfgs, slide, CBF barriers
  baselines/       samplers CS-Pi is compared against
    simple.py        unguided, projection, rejection      (black-box policy)
    guidance.py      guidance, trust                      (need policy.velocity, constraint.cost)
    cbf.py           safediffuser, safeflowmatcher        (need policy.velocity, constraint.barrier_rows)
  core/            CS-Pi and the YAML machinery
    algorithm.py     cspi()                               (the method, ~50 lines)
    registry.py      name -> object resolution for YAML fields
    run.py           runs one experiment YAML:   python -m cspi.core.run configs/<name>.yaml
    tables.py        prints its table:           python -m cspi.core.tables configs/<name>.yaml
  benchmarks/      environments: tasks, rollouts, metrics (avoid2d, spheres, orient, shelf; mpd.py = MPD glue)
  loglik.py        exact log-likelihood of samples under a flow (the logL metric)
  train.py, figures.py
configs/           one YAML per experiment (paper tables, sweeps, ablations); configs/train/ for priors
configs/examples/  minimal example experiments
examples/custom_policy.py   your own policy + constraint, in Python and through a YAML
tests/test_api.py           contract and YAML checks, no assets needed
tests/test_release.py       golden replay of the paper code (needs assets/ and MPD)
```

## 3. The policy contract (`cspi/policies/base.py`)

Time convention everywhere: **t = 0 is noise, t = 1 is data.**

| Member | Needed by | Meaning |
|---|---|---|
| `condition(obs)` | all | Returns a *new* bound policy for raw observations `obs (B, obs_dim)`. It sets `shape`, `obs_n` and resets `nfe = 0`. |
| `shape` | all | `(B, H, d)`, the sample shape. |
| `obs_n` | all | The conditioning in model space. Benchmarks store it with the samples for logL. |
| `nfe` | all | Counter of network evaluations. Every network call must increment it. |
| `normalizer` | all | `Normalizer` from model space to physical units (`Normalizer.identity()` if none). |
| `noise()` | all | One draw from the source distribution, shape `shape`. |
| `sample(step_size)` | all | Full sample, t: 0 → 1. |
| `denoise(x, t0, step_size)` | CS-Pi | Finish a sample sitting at time `t0`. |
| `renoise(x1, t, eps)` | CS-Pi | Put clean `x1` back at time `t`. `eps=None` means no noise. The default is the flow interpolant `t*x1 + (1-t)*eps`. |
| `velocity(x, t)` | guidance, trust, safediffuser, safeflowmatcher | dx/dt of the sampling ODE (counts one NFE). |
| `velocity_fn(x, t_vec, obs_n)` | logL metric | The raw network, not counted. Optional: logL is skipped without it. |

**CS-Pi only needs `sample`, `renoise` and `denoise`.** Any flow or diffusion model that can
restart from an intermediate noise level qualifies. One-step or distilled policies that cannot
denoise from an arbitrary `t` can only run `unguided`, `projection` and `rejection`.

How to plug a policy in, from least to most work:

1. **Flow-matching / rectified-flow model** with the paper's interpolant
   `x_t = (1-t) eps + t x1`: wrap the velocity function.
   ```python
   from cspi import FlowPolicy, Normalizer
   policy = FlowPolicy(lambda x, t, obs: my_net(x, t, obs), horizon=16, action_dim=2,
                       normalizer=Normalizer("meanstd", act_mean, act_std, obs_mean, obs_std))
   ```
   If your model uses the *opposite* time convention (t = 1 noise), wrap it as
   `lambda x, t, obs: -my_net(x, 1 - t, obs)`.
2. **ε-prediction diffusion model** (DDPM-trained): use
   `DDIMPolicy(eps_fn, alphas_cumprod, horizon, action_dim, normalizer)`, with
   `eps_fn(x, timestep_long, obs_n)`. Flow time maps onto the training schedule (see the
   docstring). Re-tune `k`, because the rollback depth `n` now means a different noise level.
   No velocity is exposed, so the gradient and CBF baselines are unavailable.
3. **Anything else** (other schedules, x0-prediction, consistency models with multistep
   sampling, a remote API): subclass `Policy` and implement `condition`, `noise`, `denoise`
   and, if your noise schedule is not the flow interpolant, `renoise`. Use `FlowPolicy` as the
   template (about 30 lines).

To use your policy from a YAML, write a loader `def load(..., device="cpu") -> Policy` that
returns an **unbound** policy, and reference it as
`policy: {type: "my_pkg.my_module:load", <its kwargs>}`. See `examples/custom_policy.py` and
`configs/examples/custom_policy_avoid2d.yaml`.

**Matching an existing benchmark.** A new policy can drop into an existing benchmark only if it
emits that benchmark's action representation, accepts its observation layout, and its
`normalizer` maps to the benchmark's physical units:

| Benchmark | Sample (H × d) | Physical meaning | obs |
|---|---|---|---|
| avoid2d | 16 × 2 | velocity actions (waypoints = pos + cumsum) | (x, y, goal_x, goal_y) |
| spheres | 28 × 7 | B-spline control points (joint rad) | (q_start, q_goal) |
| orient | 16 × 7 | absolute joint configurations | (q_current, q_goal) |
| shelf | 16 × 7 | absolute joint configurations | (q_current, cube_xyz) |

## 4. The constraint contract (`cspi/constraints/base.py`)

| Member | Needed by | Meaning |
|---|---|---|
| `violation(x)` | CS-Pi, projection, rejection | `(B,) >= 0`, where 0 means feasible. Compared against the sampler's `tau`. |
| `project(x, final=False)` | CS-Pi, projection | P(x). `final=True` is the terminal call (bigger budget). Must return model-space x. |
| `fallback(x)` | CS-Pi | Terminal projection. Default `project(x, final=True)`. Override when the loop's P handles soft terms the fallback should not (see `Coupled2D`). |
| `post_project(x)` | methods with `post_project: true` | Default `fallback(x)`. `ChunkCollision` uses surface sliding. |
| `cost(x)` | guidance, trust | Differentiable `(B,)` surrogate. Must be autograd-friendly in x. |
| `barrier_rows(x)` | safediffuser, safeflowmatcher | `(b (B,R), A (B,R,H*d))`: barrier values (>= 0 when safe) and their Jacobian w.r.t. flattened x. |
| `grad_mask` | guidance, trust | `(1,H,d)` mask of coordinates gradient methods may move, or None. |

Constraints are built **per decision** as `cls(bound_policy, **context, **yaml_params)`. The
context is what the benchmark knows at that moment:

| Benchmark | Context keys passed to the constraint |
|---|---|
| avoid2d | `pos (B,2)`, `obstacles [(cx, cy, r), ...]` (scenario obstacles; all obstacles for `coupled_*` units) |
| spheres | `task` (MPD PlanningTask), `cost` (MPDCost), `q_start (7,)`, `q_goal (7,)` |
| orient | `robot` (RobotPanda), `q_current (B,7)`, `ee_ref (H,3)` or None |
| shelf | `task`, `cost` (MPDCost incl. scenario boxes), `q_current (B,7)`, `boxes` |

A new constraint for an existing benchmark must accept that benchmark's context keys (extra
keys are not filtered out). The easiest way to write one is to subclass `ActionSpaceConstraint`
and implement `action_violation(a)`, `action_project(a, final)` and optionally
`action_cost(a)` on physical actions `a = normalizer.unnormalize(x)`. `Obstacles2D` is the
reference. Register it in `CONSTRAINTS` (`cspi/constraints/__init__.py`), or reference it as
`"pkg.module:Class"`.

**The projection defines CS-Pi's metric.** `delta = ||P(x) - x||` is measured in model space,
and `n = k / (k + delta)`. A projection written in physical units is wrapped by
normalize/unnormalize, so `delta` stays in model units. Doubling the normalization scale doubles
`delta`, which halves the effective `k`.

## 5. Experiment YAML schema

```yaml
# configs/<name>.yaml — results go to $CSPI_RESULTS/<name>/<variant>/<unit>.{json,pt}
benchmark: avoid2d                      # cspi/benchmarks/<name>.py, or "my_pkg.my_bench" (module path)
policy: {type: flow, checkpoint: avoid2d}            # loader name or "pkg.module:fn", + its kwargs
constraint: {type: obstacles_2d, margin: 0.01}       # class name or "pkg.module:Class", + its params
units: [left, right, both]              # what evaluate_unit iterates over (scenarios / splits / seeds)
eval: {n_seeds: 40, n_episodes: 5}      # kwargs of the benchmark's evaluate_unit
logl: true                              # score the samples' log-likelihood under the policy
logl_cap: 400                           # at most this many samples per cell are scored (null = all)
table:                                  # how cspi.core.tables prints it
  columns: {Safe-SR: success, NFE: nfe}
  percent: [success]
  per_unit: false
methods:                                # label -> sampler + hyperparameters
  CS-Pi:    {sampler: cspi, k: [10.0, 20.0], k_max: 5, tau: 0.01, step_size: 0.5}
  Guidance: {sampler: guidance, alpha: 10.0, step_size: 0.2, constraint: {guide_temperature: 0.05}}
  Trust:    {sampler: trust, w: 0.05, j_max: 5, post_project: true, eval: {n_seeds: 10}}
# methods_from: tab_d3il.yaml           # alternatively reuse another config's methods
# only: [CS-Pi, Trust]                  #   (optionally a subset)
```

Rules:

- **Any list is a sweep.** The Cartesian product over all list-valued keys of a method,
  including keys inside its `constraint:` / `eval:` overrides, makes the method's variants.
  The variant name is `<label>_<key><value>...` using the last key segment
  (e.g. `CS-Pi_k10.0`). A parameter that really is a list cannot be passed directly; wrap it in a
  dict or a string.
- A method's `constraint:` and `eval:` blocks override the experiment-level ones. That includes
  `constraint: {type: ...}`.
- `post_project: true` applies `constraint.post_project` to each sample the method returns.
- Every (variant, unit) pair is one **cell**. Cells whose JSON already exists are skipped, so
  delete a cell's results to rerun it. `--list` prints the number of cells (the SLURM array
  size) and `--cell i` runs one.
- Environment: `CSPI_ASSETS` (default `assets`), `CSPI_RESULTS` (default `results`), `MPD_ROOT`.

## 6. Hyperparameter reference

### CS-Pi (`sampler: cspi`)

| Key | Default | Meaning / how to set it |
|---|---|---|
| `k` | required | Rollback scale. Depth `n = k / (k + delta)`: for a correction of size `delta = k` the sample is renoised to `n = 0.5` (half noise). Larger `k` means shallower rollback (closer to plain projection), smaller `k` means deeper rollback (more re-generation, more NFE). Set it on the scale of a typical `delta` in model space: print `info["delta_history"]` for a few samples. Paper: 2D 20, Sphere-Avoid 50, Shelf 50, Orient 10. |
| `k_max` | required | Maximum project / renoise / re-denoise iterations. Paper: 2D 5, Spheres 20, Shelf 5, Orient 3. `k_max: 0` is plain projection. |
| `tau` | required | Feasibility threshold on `violation(x)`, in the constraint's units (2D: summed penetration depth; Panda: MPD cost). `tau: 0` with a never-zero violation runs the full `k_max` budget. |
| `step_size` | 0.1 | Euler step in flow time. A full sample takes `round(1/step_size)` steps, re-denoising from `n` takes `ceil((1-n)/step_size)`. |
| `terminal_projection` | true | Apply `constraint.fallback` if still infeasible after the loop. Turning it off executes infeasible samples as sampled (`tpoff_*` configs). |
| `n_schedule` | `adaptive` | `adaptive` = `k/(k+delta)`. A float gives a fixed depth (ablation). |
| `fresh_noise` | true | false renoises with zero noise (`x_n = n g`, ablation). |
| `n_threshold` | null | Early stop once `n` exceeds it (the correction is small enough to just project). |
| `patience`, `patience_eps_rel` | 0, 0.05 | Early stop when `n` closed less than `patience_eps_rel` of its gap to 1 over the last `patience` iterations. |

`delta` is the **batch maximum** of the correction norms. All B samples share one `n`, so one
far-off sample makes the whole batch roll back deeply. Results depend on the batch size the
benchmark uses.

### Baselines

| Sampler | Keys (default) | Meaning |
|---|---|---|
| `unguided` | `step_size` (0.1) | Prior samples, no constraint. |
| `projection` | `tau`, `step_size` (0.1) | Prior sample, then the terminal projection. |
| `rejection` | `budget`, `tau`, `step_size` (0.1) | Up to `budget` prior samples per element; first feasible, else least violating. |
| `guidance` | `alpha`, `step_size` (0.1) | Velocity corrected by `-alpha * grad cost(x1_hat)` at the Tweedie estimate, at every step but the last. |
| `trust` | `w`, `j_max`, `step_size` (0.1), `stop_tol` (1e-6), `v_max_mult` (1.5), `schedule` (`constant` / `linear`), `j_start` (0) | Trust Sampling: up to `j_max` normalized gradient steps of size `w` per Euler step, through the network. Stops when cost < `stop_tol` or `‖v‖` > `v_max_mult` × the unguided ‖v‖. |
| `safediffuser` | `n_steps` (10), `lam` (1.0), `delta` (0), `state` (`ode` / `tweedie`), `slack` (0) | CBF-QP filter on every Euler step, class-K gain `lam`, barrier offset `delta`, constant slack weight. |
| `safeflowmatcher` | `T_p` (1), `T_c` (10), `alpha` (2.0), `eps` (0.5), `rho` (0.9), `delta` (0), `state` (`ode`), `slack` (0 or `sfm`), `t_w` (0.9), `w_scale` (200) | `T_p` prediction steps, then `T_c` filtered correction steps on `alpha (1-t) v` with the finite-time barrier condition (`eps`, `rho`). `slack: sfm` uses the vanishing schedule `w_scale (1 - exp(3 (t/t_w - 1)))`. |

### Constraints (`constraint:` block)

| Type | Keys (default) | Meaning |
|---|---|---|
| `obstacles_2d` | `margin` (0.01), `guide_temperature` (0.02), `guide_margin` (0.005) | Obstacle inflation for violation and P. Sigmoid sharpness and inflation of the guidance cost. |
| `coupled_2d` | `y_ref` (0.7), `lam` (1.0), `margin` (0.005) | Obstacles (hard) plus `lam * mean_t (y_t - y_ref)^2` (soft). P = proximal step on the preference, then the obstacle projection. |
| `identity` | none | P = identity, always "violated" (SDEdit check). |
| `spline_collision` | `proj_steps` (10), `tau` (1e-3), `cbf_groups` (`obs+jl`) | Newton steps per projection (×5 terminal). Steps stop at cost ≤ `tau`. Barrier groups `obs`, `jl`, `self`. |
| `orientation_lock` | `lam_orient` (10), `lam_ee` (3000), `lam_jl` (100), `beta` (1e6), `proj_steps` (5), `cbf_groups` (`orient+jl`) | Orientation weight, EE-proximity weight (active when the benchmark supplies `ee_ref`), joint-limit weight in the guidance cost, proximal weight of P (large = trust the cost), L-BFGS iterations (×5 terminal). |
| `chunk_collision` | `proj_steps` (6), `beta` (1.0), `tau` (1e-3), `cbf_groups` (`obs+jl`) | L-BFGS iterations of the proximal P (×5 terminal and for sliding), proximal weight, sliding stop threshold, barrier groups. |

### Benchmarks (`eval:` block)

| Benchmark | Keys (default) | Units |
|---|---|---|
| avoid2d | `n_seeds` (40), `n_episodes` (5) | scenario names, or `coupled_s<seed>` (soft + hard composition: one batch of `n_episodes` chunks; needs `coupled_2d`) |
| spheres | `batch` (8), `tau` (1e-3, success threshold), `seed` (0), `n_tasks` (all 120) | `id`, `ood` |
| orient | `ee_prox` (false: compute `ee_ref` from one unguided rollout), `n_tasks` (all 30) | task seeds `0, 1, 2` |
| shelf | `seed` (42), `tau` (1e-3, success threshold), `n_tasks` (all 30) | scenario names |

## 7. Recipes

**Run CS-Pi on your policy in plain Python** (no benchmark, no YAML):

```python
from cspi import FlowPolicy, ActionSpaceConstraint, cspi
policy = FlowPolicy(my_velocity_fn, horizon=H, action_dim=d, normalizer=my_norm).condition(obs)
x, info = cspi(policy, MyConstraint(policy, ...), k=20.0, k_max=5, tau=1e-3, step_size=0.1)
actions = policy.normalizer.unnormalize(x)
```

`python examples/custom_policy.py` runs a complete toy example.

**Evaluate your policy on a paper benchmark:** write a loader, copy the paper config (e.g.
`configs/tab_d3il.yaml`), replace its `policy:` line, keep or re-tune the method settings, then
run `python -m cspi.core.run configs/<yours>.yaml` and `python -m cspi.core.tables configs/<yours>.yaml`.

**Add a constraint:** subclass `Constraint` or `ActionSpaceConstraint` and accept the benchmark's
context keys. Add `cost` / `barrier_rows` only if you want to run the gradient / CBF baselines.
Register it in `CONSTRAINTS` or use the `"pkg.module:Class"` form.

**Add a baseline:** write `def my_sampler(policy, constraint, **hp) -> (x, dict(nfe=...))`.
Measure NFE as `policy.nfe` after minus before. Register it in `BASELINES`
(`cspi/baselines/__init__.py`) or use `sampler: "pkg.module:my_sampler"`.

**Add a benchmark:** a module with

```python
def evaluate_unit(unit, sampler, params, policy, make_constraint, assets, **eval_kwargs) -> list[dict]
def summarize(records) -> dict[str, float]
```

At each decision, call `p = policy.condition(obs)`, `c = make_constraint(p, **context)` and
`x, info = sampler(p, c, **params)`, then execute `p.normalizer.unnormalize(x)`. Each record
should keep `x` (model-space samples) and `obs` (`p.obs_n`) for the logL metric, plus `nfe`.
`score=False` excludes a record from logL. Use `avoid2d.py` as the template.

## 8. Gotchas

- **Panda benchmarks are CPU-only.** The FK library (torchkin) keeps internal CPU tensors.
  `cspi/benchmarks/mpd.py` must be imported before anything from `torch_robotics` (it puts MPD on
  `sys.path` and patches a torchkin memory leak). Constraints receive MPD objects and never
  import MPD themselves.
- **NFE is counted by the policy.** Call the network only through `velocity` / `denoise` /
  `sample`. A direct `velocity_fn` call is invisible to the NFE metric.
- **`condition` returns a fresh object.** Do not mutate a bound policy across decisions. Bind a
  new one per decision, as the benchmarks do.
- **RNG order is part of the result.** The golden tests compare against the paper code
  sample for sample. Adding or reordering `torch.randn` calls in a sampler or benchmark changes
  every number. Run `tests/test_release.py` after touching anything in `core/`, `baselines/`,
  `constraints/` or `benchmarks/`.
- **`tau` is in the constraint's units**, and the sampler's `tau` (when to stop iterating) is
  separate from a constraint's or benchmark's own `tau` (projection stop / success threshold).
- **Lists in YAML are sweeps** (see §5).
- **Finished cells are skipped.** Delete `results/<config>/<variant>/<unit>.json` to rerun one.

## 9. Checks

```bash
python -m pytest tests/test_api.py -q        # contracts, toy flow + DDIM policies, YAML expansion (seconds, no assets)
python -m pytest tests/ -q                   # + golden replay of every benchmark (needs assets/ and MPD, ~1.5 min)
python examples/custom_policy.py             # toy end-to-end
python -m cspi.core.run configs/examples/custom_policy_avoid2d.yaml   # custom policy through the YAML runner
```
