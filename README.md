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

The whole method is `cspi()` in [`cspi/samplers.py`](cspi/samplers.py) (about 40 lines). Every
baseline in the paper (unguided, projection, rejection, gradient guidance, Trust Sampling,
SafeDiffuser, SafeFlowMatcher) is written once against the same small `Problem` interface, and each
benchmark only supplies its prior, its constraint and its projection `P`.

## Repository layout

```
cspi/
  flow.py          flow-matching networks, normalizer, OT-CFM loss, Euler solver, checkpoint I/O
  samplers.py      Problem interface; CS-Pi, unguided, projection, rejection, guidance, Trust Sampling
  cbf.py           SafeDiffuser and SafeFlowMatcher (CBF-QP safety filters) and their QP solver
  loglik.py        exact log-likelihood of a sample under the prior (the "logL" columns)
  benchmarks/
    avoid2d.py     2D Avoidance: point robot, 6 obstacle scenarios, support metric
    mpd.py         shared Franka Panda glue on MPD: planning task, collision cost, projections, rollout
    spheres.py     Sphere-Avoid: whole-trajectory planning around 10 / 16 spheres
    orient.py      3D-Orient: reach the goal with a fixed end-effector orientation
    shelf.py       Shelf pick-and-place with unseen obstacle boxes
  run.py           runs one config (one paper table / figure): python -m cspi.run configs/<name>.yaml
  tables.py        prints the table of a config:              python -m cspi.tables configs/<name>.yaml
  figures.py       analyses of saved rollouts (n dynamics, fallback, smoothness)
  train.py         trains a prior:                            python -m cspi.train configs/train/<name>.yaml
configs/           one YAML per table / figure (methods x units, sweeps as lists), configs/train/ for priors
eval.sh            runs every inference experiment (locally or as SLURM arrays), then prints all tables
train.sh           trains the four priors
scripts/download_assets.py   checkpoints, evaluation tasks and training data from the Hugging Face Hub
third_party/mpd.patch        four small fixes to MPD needed by the Panda benchmarks
```

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

**Metrics.** Safe-SR counts rollouts that complete the task without violating the constraint; NFE
is the number of network evaluations per rollout; logL is the mean log-likelihood the prior assigns
to the method's samples, per dimension (higher is more in-distribution), from 24 backward Euler steps
with a 6-probe Hutchinson divergence estimate. All rows of a table come from one run of its config.

## Training the priors

The released checkpoints are the ones used in the paper. To retrain them:

```bash
bash train.sh              # all four priors (GPU recommended); bash train.sh slurm to submit
```
