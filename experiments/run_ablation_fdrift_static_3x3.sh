#!/usr/bin/env bash
set -euo pipefail

# Single-factor batch: force/torque bias drift only.
# Keep perturb_intensity fixed and only change force/torque drift random-walk params.

cd "$(dirname "$0")/.."
export PYTHONPATH=.
PYTHON_BIN=${PYTHON_BIN:-/home/sun/anaconda3/envs/pih_env/bin/python}

TIMESTEPS=200000
PERTURB_INTENSITY=0.65
OBS_MODE="vision-touch"
SEED=7

# Baseline config in pih_rebuild/config.py:
# force_bias_drift_std=0.3, force_bias_drift_max=6.0
# torque_bias_drift_std=0.03, torque_bias_drift_max=0.6
# Sweep uses half / default / double.
LEVELS=("low" "mid" "high")
F_DRIFT_STDS=("0.15" "0.30" "0.60")
F_DRIFT_MAXS=("3.0" "6.0" "12.0")
T_DRIFT_STDS=("0.015" "0.03" "0.06")
T_DRIFT_MAXS=("0.3" "0.6" "1.2")

for i in "${!LEVELS[@]}"; do
  level="${LEVELS[$i]}"
  f_std="${F_DRIFT_STDS[$i]}"
  f_max="${F_DRIFT_MAXS[$i]}"
  t_std="${T_DRIFT_STDS[$i]}"
  t_max="${T_DRIFT_MAXS[$i]}"

  tag="rebuild_fdrift_static_${level}"
  echo "=== Running level=${level}, seed=${SEED}, f_std=${f_std}, f_max=${f_max}, t_std=${t_std}, t_max=${t_max} ==="

  PYTHONWARNINGS=ignore "$PYTHON_BIN" -m pih_rebuild.train_sac \
    --obs_mode "$OBS_MODE" \
    --perturb \
    --perturb_intensity "$PERTURB_INTENSITY" \
    --force_bias_drift_std "$f_std" \
    --force_bias_drift_max "$f_max" \
    --torque_bias_drift_std "$t_std" \
    --torque_bias_drift_max "$t_max" \
    --timesteps "$TIMESTEPS" \
    --seed "$SEED" \
    --tag "$tag" 2>&1 | \
    grep --line-buffered -E "^\[|success_rate_window|ep_rew_mean|ep_len_mean|total_timesteps|final_window_success_rate|best_window_success_rate|best_eval_success_rate|saved_model|F_bias"
done

echo "All force-drift ablation runs finished."
