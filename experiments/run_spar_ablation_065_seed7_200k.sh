#!/usr/bin/env bash
# SPAR-SAC ablation @ perturb 0.65, seed 7, 200k steps, vision-touch, h220.
# Runs the current three-arm comparison: SAC, SAC+M1, SAC+M1+M2.
set -euo pipefail

PY=/home/sun/anaconda3/envs/pih_env/bin/python
cd "$(dirname "$0")/.."

COMMON=(--obs_mode vision-touch --max_steps 220 --perturb --perturb_intensity 0.65 \
        --timesteps 200000 --seed 7 --eval_freq 25000 --eval_seeds 20 \
        --learning_starts 1000 --batch_size 256 --learning_rate 3e-4 --ent_coef auto_0.2)

echo "=== SAC ==="
PYTHONWARNINGS=ignore "$PY" -m pih_rebuild.train_sac --algo sac --tag spar_sac "${COMMON[@]}"

echo "=== SAC + M1 ==="
PYTHONWARNINGS=ignore "$PY" -m pih_rebuild.train_sac --algo m1 --tag spar_m1 "${COMMON[@]}"

echo "=== SAC + M1 + M2 ==="
PYTHONWARNINGS=ignore "$PY" -m pih_rebuild.train_sac --algo m1m2 --tag spar_m1m2 "${COMMON[@]}"
