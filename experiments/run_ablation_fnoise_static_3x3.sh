#!/usr/bin/env bash
set -euo pipefail

# Single-factor batch: per-step force/torque Gaussian noise only.
# Keep perturb_intensity=0.65 fixed; only sweep force_noise_std / torque_noise_std
# (half / default / double together, since they are coupled by the same intensity multiplier).
# Config defaults: force_noise_std = 2.0 N,  torque_noise_std = 0.2 N*m

cd "$(dirname "$0")/.."
export PYTHONPATH=.
PYTHON_BIN=${PYTHON_BIN:-/home/sun/anaconda3/envs/pih_env/bin/python}

TIMESTEPS=200000
PERTURB_INTENSITY=0.65
OBS_MODE="vision-touch"
SEED=7

# low  = 0.5×  force=1.0 N   torque=0.10 N*m
# mid  = 1.0×  force=2.0 N   torque=0.20 N*m  (baseline default)
# high = 2.0×  force=4.0 N   torque=0.40 N*m
LEVELS=("low" "mid" "high")
FORCE_STDS=("1.0" "2.0" "4.0")
TORQUE_STDS=("0.1" "0.2" "0.4")

for i in "${!LEVELS[@]}"; do
  level="${LEVELS[$i]}"
  f_std="${FORCE_STDS[$i]}"
  t_std="${TORQUE_STDS[$i]}"

  tag="rebuild_fnoise_static_${level}"
  echo "=== Running level=${level}, seed=${SEED}, force_noise_std=${f_std}, torque_noise_std=${t_std} ==="

  PYTHONWARNINGS=ignore "$PYTHON_BIN" -m pih_rebuild.train_sac \
    --obs_mode "$OBS_MODE" \
    --perturb \
    --perturb_intensity "$PERTURB_INTENSITY" \
    --force_noise_std "$f_std" \
    --torque_noise_std "$t_std" \
    --timesteps "$TIMESTEPS" \
    --seed "$SEED" \
    --tag "$tag" 2>&1 | \
    grep --line-buffered -E "^\[|success_rate_window|ep_rew_mean|ep_len_mean|total_timesteps|final_window_success_rate|best_window_success_rate|best_eval_success_rate|saved_model|vis_bias|F_bias"
done

echo "All force-noise ablation runs finished."
