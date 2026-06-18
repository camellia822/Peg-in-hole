#!/usr/bin/env bash
# 扰动强度扫描：用法 bash run_sac_intensity_seed7_200k.sh <perturb_intensity>
# 例: bash experiments/run_sac_intensity_seed7_200k.sh 0.35
set -euo pipefail
P=${1:?需要扰动强度参数, 例如 0.35}
PY=/home/sun/anaconda3/envs/pih_env/bin/python
cd "$(dirname "$0")/.."
PTAG=${P/./}   # 0.35 -> 035
LOG_FREQ=${LOG_FREQ:-1000}
SUCCESS_WINDOW=${SUCCESS_WINDOW:-20}
STATS_WINDOW_SIZE=${STATS_WINDOW_SIZE:-100}
TB_LOG_INTERVAL=${TB_LOG_INTERVAL:-4}
MAX_STEPS=${MAX_STEPS:-200}
PYTHONWARNINGS=ignore "$PY" -m pih_rebuild.train_sac \
  --tag "sac_p${PTAG}_seed7" \
  --obs_mode vision-touch \
  --max_steps "$MAX_STEPS" \
  --perturb --perturb_intensity "$P" \
  --timesteps 200000 \
  --seed 7 \
  --log_freq "$LOG_FREQ" \
  --success_window "$SUCCESS_WINDOW" \
  --stats_window_size "$STATS_WINDOW_SIZE" \
  --tb_log_interval "$TB_LOG_INTERVAL" \
  --eval_freq 25000 --eval_seeds 20 \
  --learning_starts 1000 --batch_size 256 \
  --learning_rate 3e-4 --ent_coef auto_0.2
