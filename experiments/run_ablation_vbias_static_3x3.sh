#!/usr/bin/env bash
set -euo pipefail

# Single-factor batch: static vision bias only.
# Keep perturb_intensity fixed and only change vision_bias_xy_std / vision_bias_z_std.

cd "$(dirname "$0")/.."
export PYTHONPATH=.
PYTHON_BIN=${PYTHON_BIN:-/home/sun/anaconda3/envs/pih_env/bin/python}

TIMESTEPS=200000
PERTURB_INTENSITY=0.65
OBS_MODE="vision-touch"

# Baseline config in pih_rebuild/config.py:
# vision_bias_xy_std=0.0015, vision_bias_z_std=0.0008
# Sweep uses half / default / double.
LEVELS=("low" "mid" "high")
XY_STDS=("0.00075" "0.0015" "0.0030")
Z_STDS=("0.0004" "0.0008" "0.0016")
SEED=7

for i in "${!LEVELS[@]}"; do
  level="${LEVELS[$i]}"
  xy_std="${XY_STDS[$i]}"
  z_std="${Z_STDS[$i]}"

  tag="rebuild_vbias_static_${level}"
  echo "=== Running level=${level}, seed=${SEED}, xy_std=${xy_std}, z_std=${z_std} ==="

  PYTHONWARNINGS=ignore "$PYTHON_BIN" -m pih_rebuild.train_sac \
    --obs_mode "$OBS_MODE" \
    --perturb \
    --perturb_intensity "$PERTURB_INTENSITY" \
    --vision_bias_xy_std "$xy_std" \
    --vision_bias_z_std "$z_std" \
    --timesteps "$TIMESTEPS" \
    --seed "$SEED" \
    --tag "$tag" 2>&1 | \
    grep --line-buffered -E "^\[|success_rate_window|ep_rew_mean|ep_len_mean|total_timesteps|final_window_success_rate|best_window_success_rate|saved_model"
done

echo "All static-vision-bias ablation runs finished."
