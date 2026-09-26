#!/bin/bash
# Train the four flow-matching priors used in the paper (released checkpoints are downloaded by
# scripts/download_assets.py; retraining is only needed to reproduce the priors themselves).
#
#   bash train.sh                     # all four, locally
#   bash train.sh slurm               # one GPU job per prior
#   bash train.sh local avoid2d       # a subset (names in configs/train/)
#
# Environment: PYTHON (default: python), PARTITION / TIME for SLURM.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=${PYTHON:-python}
MODE=${1:-local}
shift || true
PRIORS=(avoid2d spheres orient shelf)
[ $# -gt 0 ] && PRIORS=("$@")

for p in "${PRIORS[@]}"; do
  if [ "$MODE" = slurm ]; then
    mkdir -p logs
    opts=(--job-name="train_$p" --output="logs/%x_%j.out" --gres=gpu:1 --cpus-per-task=8 --time="${TIME:-48:00:00}")
    [ -n "${PARTITION:-}" ] && opts+=(--partition="$PARTITION")
    sbatch "${opts[@]}" --wrap="$PYTHON -m cspi.train configs/train/$p.yaml"
  else
    $PYTHON -m cspi.train "configs/train/$p.yaml"
  fi
done
