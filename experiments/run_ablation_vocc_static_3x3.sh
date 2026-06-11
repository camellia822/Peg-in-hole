#!/usr/bin/env bash
# 消融实验：视觉遮挡触发阈值（vision_occlusion_force / vision_occlusion_depth）
#
# 三档阈值：
#   low  : force= 8N,  depth=0.001m(1mm)  → 几乎一接触就失去视觉，最难
#   mid  : force=20N,  depth=0.005m(5mm)  → 中等插入深度后遮挡
#   high : force=60N,  depth=0.015m(15mm) → 深度插入/大力才遮挡，最容易
#
# 用法:
#   bash experiments/run_ablation_vocc_static_3x3.sh      # CPU
#   bash experiments/run_ablation_vocc_static_3x3.sh 0    # GPU 0

set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=.

STEPS=200000
SEED=7
GPU=${1:-""}
PYTHON_BIN=${PYTHON_BIN:-/home/sun/anaconda3/envs/pih_env/bin/python}

run_one() {
    local level=$1
    local focc=$2
    local docc=$3
    echo "=== Running level=${level}, seed=${SEED}, vision_occlusion_force=${focc}, vision_occlusion_depth=${docc} ==="

    CMD="$PYTHON_BIN -m pih_rebuild.train_sac \
        --tag rebuild_vocc_static_${level} \
        --timesteps ${STEPS} \
        --seed ${SEED} \
        --obs_mode vision-touch \
        --max_steps 220 \
        --perturb \
        --perturb_intensity 0.65 \
        --vision_occlusion_force ${focc} \
        --vision_occlusion_depth ${docc}"

    if [ -n "$GPU" ]; then
        PYTHONWARNINGS=ignore CUDA_VISIBLE_DEVICES=$GPU $CMD
    else
        PYTHONWARNINGS=ignore $CMD
    fi
}

# low 阈值：最早遮挡，强迫策略从一开始就依赖力/力矩
run_one "low"  8.0  0.001

# mid 阈值：中等遮挡
run_one "mid"  20.0  0.005

# high 阈值：最晚遮挡，策略可以长期依赖视觉
run_one "high" 60.0  0.015

echo "All vision-occlusion ablation runs finished."
