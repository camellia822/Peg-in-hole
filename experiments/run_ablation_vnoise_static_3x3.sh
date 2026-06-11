#!/usr/bin/env bash
set -euo pipefail

# Single-factor batch: per-step vision Gaussian noise only.
# Keep perturb_intensity=0.65 fixed; only sweep vision_noise_std (half/default/double).
# Config default: vision_noise_std = 0.0005 m

cd "$(dirname "$0")/.."
export PYTHONPATH=.
PYTHON_BIN=${PYTHON_BIN:-/home/sun/anaconda3/envs/pih_env/bin/python}

TIMESTEPS=200000
PERTURB_INTENSITY=0.65
OBS_MODE="vision-touch"
SEED=7

# low  = 0.5×  = 0.00025
# mid  = 1.0×  = 0.00050  (baseline default, serves as reference)
# high = 2.0×  = 0.00100
LEVELS=("low" "mid" "high")
VNOISE_STDS=("0.00025" "0.00050" "0.00100")

for i in "${!LEVELS[@]}"; do
  level="${LEVELS[$i]}"
  v_std="${VNOISE_STDS[$i]}"

  tag="rebuild_vnoise_static_${level}"
  echo "=== Running level=${level}, seed=${SEED}, vision_noise_std=${v_std} ==="

  PYTHONWARNINGS=ignore "$PYTHON_BIN" -m pih_rebuild.train_sac \
    --obs_mode "$OBS_MODE" \
    --perturb \
    --perturb_intensity "$PERTURB_INTENSITY" \
    --vision_noise_std "$v_std" \
    --timesteps "$TIMESTEPS" \
    --seed "$SEED" \
    --tag "$tag" 2>&1 | \
    grep --line-buffered -E "^\[|success_rate_window|ep_rew_mean|ep_len_mean|total_timesteps|final_window_success_rate|best_window_success_rate|best_eval_success_rate|saved_model|vis_bias|F_bias"
done

echo "All vision-noise ablation runs finished."
