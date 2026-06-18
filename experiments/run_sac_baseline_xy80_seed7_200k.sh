#!/usr/bin/env bash
# Clean SAC baseline with harder initial XY randomization: +/-80mm.
# Lean v4 reward, NO perception perturbation, vision-touch (27D), seed 7, 200k steps.
#
# Usage: bash experiments/run_sac_baseline_xy80_seed7_200k.sh
set -euo pipefail
PY=/home/sun/anaconda3/envs/pih_env/bin/python
cd "$(dirname "$0")/.."

LOG_FREQ=${LOG_FREQ:-1000}
SUCCESS_WINDOW=${SUCCESS_WINDOW:-20}
STATS_WINDOW_SIZE=${STATS_WINDOW_SIZE:-100}
TB_LOG_INTERVAL=${TB_LOG_INTERVAL:-4}
MAX_STEPS=${MAX_STEPS:-200}
INIT_XY_RANDOM_MM=${INIT_XY_RANDOM_MM:-80}

echo "=== SAC clean baseline XY +/-${INIT_XY_RANDOM_MM}mm (lean v4 reward, no perturb) ==="
PYTHONWARNINGS=ignore "$PY" -m pih_rebuild.train_sac \
  --tag "sac_xy${INIT_XY_RANDOM_MM}_clean_seed7" \
  --obs_mode vision-touch \
  --max_steps "$MAX_STEPS" \
  --init_xy_random_mm "$INIT_XY_RANDOM_MM" \
  --timesteps 200000 \
  --seed 7 \
  --log_freq "$LOG_FREQ" \
  --success_window "$SUCCESS_WINDOW" \
  --stats_window_size "$STATS_WINDOW_SIZE" \
  --tb_log_interval "$TB_LOG_INTERVAL" \
  --eval_freq 25000 --eval_seeds 20 \
  --learning_starts 1000 --batch_size 256 \
  --learning_rate 3e-4 --ent_coef auto_0.2
