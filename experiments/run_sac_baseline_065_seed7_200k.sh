#!/usr/bin/env bash
# SAC baseline @ perturb 0.65, seed 7, 200k steps, vision-touch, h220.
set -euo pipefail

PY=/home/sun/anaconda3/envs/pih_env/bin/python
cd "$(dirname "$0")/.."

COMMON=(--obs_mode vision-touch --max_steps 220 --perturb --perturb_intensity 0.65 \
        --timesteps 200000 --seed 7 --eval_freq 25000 --eval_seeds 20 \
        --learning_starts 1000 --batch_size 256 --learning_rate 3e-4 --ent_coef auto_0.2)

echo "=== SAC ==="
PYTHONWARNINGS=ignore "$PY" -m pih_rebuild.train_sac --tag sac_p065_seed7 "${COMMON[@]}"
