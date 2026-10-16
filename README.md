# CS-Pi: constraint-satisfying sampling from a frozen flow policy

CS-Pi samples from a pretrained flow-matching policy so that the sample satisfies a constraint the
policy never saw in training, while staying close to what the policy would have done. It needs no
retraining and no gradient of the constraint, only a projection onto the constraint set:

```
x   <- ODE(noise, t: 0 -> 1)                  sample from the prior
while x violates the constraint (at most K_max times):
    g   <- P(x)                                project onto the constraint set
    n   <- k / (k + ||g - x||)                 rollback depth: a large correction rolls back further
    x_n <- n g + (1 - n) eps                   renoise the projection to flow time n
    x   <- ODE(x_n, t: n -> 1)                 let the prior re-denoise it
if x still violates the constraint: x <- P(x)  terminal projection
```

The whole method is `cspi()` in [`cspi/core/algorithm.py`](cspi/core/algorithm.py) (about 50 lines). It takes
a **policy** and a **constraint** through two small contracts, so it runs on any flow or diffusion policy
that can restart sampling from an intermediate noise level. Every baseline in the paper (unguided,
projection, rejection, gradient guidance, Trust Sampling, SafeDiffuser, SafeFlowMatcher) is written
against the same contracts, and every experiment is one YAML file that names its benchmark, policy,
constraint and methods. **To use CS-Pi with your own policy or constraint, read
[`FOR_AGENTS.md`](FOR_AGENTS.md)** (written for both AI agents and people).

```python
from cspi import FlowPolicy, cspi
policy = FlowPolicy(my_velocity_fn, horizon=16, action_dim=2, normalizer=my_norm).condition(obs)
x, info = cspi(policy, my_constraint(policy), k=20.0, k_max=5, tau=0.01)
```

## Repository layout

```
cspi/
  policies/        policy contract (base.py); FlowPolicy, DDIMPolicy adapter, networks, normalizer
  constraints/     constraint contract (base.py); the paper's constraints and projections
  baselines/       unguided / projection / rejection, guidance / Trust Sampling, SafeDiffuser / SafeFlowMatcher
  core/            CS-Pi (algorithm.py), YAML runner (run.py), tables (tables.py), name registry
  benchmarks/      environments, tasks, rollouts and metrics: avoid2d, spheres, orient, shelf (+ mpd.py glue)
  loglik.py        exact log-likelihood of a sample under the prior (the "logL" columns)
  figures.py       analyses of saved rollouts (n dynamics, fallback, smoothness)
  train.py         trains a prior:                            python -m cspi.train configs/train/<name>.yaml
configs/           one YAML per experiment (paper table / figure / sweep), configs/train/ for priors
eval.sh            runs every inference experiment (locally or as SLURM arrays), then prints all tables
train.sh           trains the four priors
examples/          your own policy and constraint, in Python and through a YAML
FOR_AGENTS.md      contracts, YAML schema, every hyperparameter, recipes for new policies / constraints / benchmarks
scripts/download_assets.py   checkpoints, evaluation tasks and training data from the Hugging Face Hub
third_party/mpd.patch        four small fixes to MPD needed by the Panda benchmarks
```

One experiment by hand: `python -m cspi.core.run configs/<name>.yaml` (add `--list` / `--cell i` for one
cell), then `python -m cspi.core.tables configs/<name>.yaml`.

## Setup

```bash
conda create -n cspi python=3.8 && conda activate cspi
pip install torch==2.0.0 && pip install -r requirements.txt

# MPD (robot model, environments, collision fields) for the three Panda benchmarks
git clone https://github.com/joaoamcarvalho/mpd-splines-public && cd mpd-splines-public
git checkout 3676cbf && git apply ../cspi/third_party/mpd.patch   # then follow MPD's install instructions
export MPD_ROOT=$PWD && cd -

python scripts/download_assets.py      # add --no-train-data to skip the training sets
```

The 2D benchmark needs only PyTorch; `MPD_ROOT` is only read by the Panda benchmarks.

## Reproducing the paper

```bash
bash eval.sh               # every experiment, locally
bash eval.sh slurm         # one SLURM array per config; set PARTITION, CPUS, TIME, MAX_PAR as needed
bash eval.sh tables        # print all tables from results/
bash eval.sh local tab_shelf          # a single table
```

Each config writes `results/<config>/<method>/<unit>.json` (metrics) and `.pt` (the rollouts).
Hyperparameter sweeps run first because they select the settings used in the tables.

| Paper | Config | What it runs |
|---|---|---|
| Table: Shelf (`tab:shelf`) | `configs/tab_shelf.yaml` | 8 methods x 5 scenarios x 30 tasks |
| Table: 2D Avoidance (`tab:d3il-main`) | `configs/tab_d3il.yaml` | 8 methods x 6 scenarios x 200 rollouts |
| Table: 3D-Orient (`tab:mpd9`) | `configs/tab_orient.yaml` | 7 rows x 90 rollouts |
| Fig: soft + hard composition (`softhard-2d`) | `configs/fig_softhard_2d.yaml` | CS-Pi k sweep vs guidance / Trust |
| Table: Sphere-Avoid (`tab:spheres`) | `configs/tab_spheres.yaml` | 8 methods, ID and OOD |
| Tables: hyperparameters (`tab:hp-*`) | `configs/hp_{d3il,spheres,shelf,orient}.yaml` | sweeps |
| Table: early stopping (`tab:earlystop`) | `configs/earlystop_d3il.yaml` | 4 stopping rules x 11 values of k |
| Table: rollback schedule (`tab:schedule-ablation`) | `configs/schedule_d3il.yaml` | adaptive vs fixed depth |
| SDEdit mechanism | `configs/sdedit_d3il.yaml` | renoise / re-denoise without a constraint |
| Table: wall-clock (`tab:walltime`) | `configs/walltime_*.yaml` | timing, one process per benchmark |
| Table: n and projection error (`tab:nproj`) | `python -m cspi.figures ndynamics` | from the CS-Pi runs above |
| Table: terminal projection (`tab:fb`) | `python -m cspi.figures fallback` | from the table runs |
| Fig: smoothness (`jerk`), successful-rollout logL | `python -m cspi.figures smoothness` / `success_logl` | from the table runs |

The hardware experiments (Franka + pi0.5) are not part of this release.

**Protocol.** All rows of a table come from one run of its config. Every method that has a tunable
setting uses the value with the best Safe-SR in the matching `hp_*` sweep (ties go to fewer network
evaluations); the chosen values are written in the table configs.

**Metrics.** Safe-SR counts rollouts that complete the task without violating the constraint; NFE
is the number of network evaluations per rollout; logL is the mean log-likelihood the prior assigns
to the method's samples, per dimension (higher is more in-distribution), from 24 backward Euler steps
with a 6-probe Hutchinson divergence estimate.

**Compute.** The Panda benchmarks run their kinematics on CPU. A 2D cell takes minutes, a Shelf cell
up to two hours (Rejection); `bash eval.sh slurm` runs every cell in parallel. Give Sphere-Avoid cells
about 64 GB of memory (`MEM=64G`).

## Training the priors

The released checkpoints are the ones used in the paper. To retrain them:

```bash
bash train.sh              # all four priors (GPU recommended); bash train.sh slurm to submit
```

Training data (downloaded with the assets): the 80 hand-drawn 2D demonstrations; 10k RRT-Connect
trajectories in the 10- and 16-sphere scenes (Sphere-Avoid); MPD's EnvSpheres3D trajectories for
10k tasks (3D-Orient); and 106k multimodal pick-and-place demonstrations from a parametric-primitive
generator (Shelf, regenerated with the settings of the paper's dataset).
