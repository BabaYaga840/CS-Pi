#!/bin/bash
# Run every inference experiment in the paper, then print its tables.
#
#   bash eval.sh                          # everything, locally, one cell after another
#   bash eval.sh slurm                    # one SLURM array per config (one task per cell)
#   bash eval.sh local tab_shelf hp_d3il  # only some configs (names in configs/, without .yaml)
#   bash eval.sh tables                   # just print the tables from results/
#
# Environment: PYTHON (default: python); PARTITION, CPUS, MEM, TIME, MAX_PAR for SLURM;
# CSPI_ASSETS (default: assets), CSPI_RESULTS (default: results), CSPI_LOGS (default: logs); MPD_ROOT (see README).
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=${PYTHON:-python}
MODE=${1:-local}
shift || true

# Hyperparameter sweeps first (they select the settings in the table configs), then the tables,
# then mechanism / ablation studies. Wall-clock configs always run as one job on one node.
CONFIGS=(hp_d3il hp_spheres hp_shelf hp_orient
         tab_d3il tab_spheres tab_shelf tab_orient
         fig_softhard_2d earlystop_d3il schedule_d3il sdedit_d3il
         walltime_d3il walltime_spheres walltime_shelf walltime_orient)
[ $# -gt 0 ] && CONFIGS=("$@")

tables() {
  for c in "${CONFIGS[@]}"; do
    echo; echo "## $c"; $PYTHON -m cspi.core.tables "configs/$c.yaml"
  done
  $PYTHON -m cspi.figures all
}

case "$MODE" in
  local)
    for c in "${CONFIGS[@]}"; do $PYTHON -m cspi.core.run "configs/$c.yaml"; done
    tables ;;
  slurm)
    LOGS=${CSPI_LOGS:-logs}
    mkdir -p "$LOGS"
    for c in "${CONFIGS[@]}"; do
      n=$($PYTHON -m cspi.core.run "configs/$c.yaml" --list)
      opts=(--job-name="$c" --output="$LOGS/%x_%A_%a.out" --cpus-per-task="${CPUS:-8}" --time="${TIME:-24:00:00}")
      [ -n "${PARTITION:-}" ] && opts+=(--partition="$PARTITION")
      [ -n "${MEM:-}" ] && opts+=(--mem="$MEM")
      if [[ $c == walltime_* ]]; then
        sbatch "${opts[@]}" --output="$LOGS/%x_%j.out" --wrap="$PYTHON -m cspi.core.run configs/$c.yaml"
      else
        sbatch "${opts[@]}" --array="0-$((n - 1))%${MAX_PAR:-64}" \
          --wrap="$PYTHON -m cspi.core.run configs/$c.yaml --cell \$SLURM_ARRAY_TASK_ID"
      fi
    done
    echo "Submitted. When the jobs finish: bash eval.sh tables" ;;
  tables)
    tables ;;
  *)
    echo "usage: bash eval.sh [local|slurm|tables] [config ...]"; exit 1 ;;
esac
