#!/usr/bin/env bash
# Perception-perturbation difficulty sweep for the rebuilt dual-peg task.
# Dual hole, 1mm single-side clearance (config defaults), vision+touch fusion.
# Each run only corrupts the observation; success/reward use the true state.
set -euo pipefail

PY=${PY:-/home/sun/anaconda3/envs/pih_env/bin/python}
cd "$(dirname "$0")/.."
export PYTHONPATH=.

TIMESTEPS=${TIMESTEPS:-200000}
SEED=${SEED:-7}
TAG=${TAG:-rebuild_perturb}

for INTENSITY in 0.5 1.0 1.5 2.0; do
  echo "=== perturb_intensity=${INTENSITY} seed=${SEED} ==="
  PYTHONWARNINGS=ignore "$PY" -m pih_rebuild.train_sac \
    --obs_mode vision-touch \
    --perturb \
    --perturb_intensity "${INTENSITY}" \
    --timesteps "${TIMESTEPS}" \
    --seed "${SEED}" \
    --tag "${TAG}"
done
