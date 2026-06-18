#!/usr/bin/env python3
"""可视化 + 成功率评估脚本

用法示例:
  # 有训练模型 + 开启可视化窗口（默认 5 个 episode）:
  python -m pih_rebuild.eval_viz --model <path.zip>

  # 关闭可视化，批量跑 50 个 episode 评估成功率:
  python -m pih_rebuild.eval_viz --model <path.zip> --episodes 50 --no_render

  # 不加载模型，用随机策略跑 3 个 episode 看环境是否正常工作:
  python -m pih_rebuild.eval_viz --episodes 3

  # 加载默认 best_eval_model 直接运行:
  python -m pih_rebuild.eval_viz --best
"""

import argparse
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from pih_rebuild.config import DualPegTaskConfig
from pih_rebuild.envs.ur5_dual_peg_env import UR5DualPegEnv

# 项目根目录下最新的 best_eval_model（如果存在）
_DEFAULT_BEST_PATTERN = (
    "output/rebuild/cyl_dual_1mm_side_d20mm_initxy50mm_reward_v3_touch_"
    "vision-touch_h200_steps200000_seed7/model/best_eval_model.zip"
)


def parse_args():
    p = argparse.ArgumentParser(description="Dual-Peg 可视化与成功率评估")
    p.add_argument("--model", type=str, default=None,
                   help="SB3 SAC 模型路径 (.zip)")
    p.add_argument("--best", action="store_true",
                   help=f"快捷方式：加载 {_DEFAULT_BEST_PATTERN}")
    p.add_argument("--episodes", type=int, default=5,
                   help="评估 episode 数量（默认 5）")
    p.add_argument("--seed_base", type=int, default=0,
                   help="episode i 的 seed = seed_base + i")
    p.add_argument("--no_render", action="store_true",
                   help="关闭可视化，仅输出统计结果")
    p.add_argument("--sleep", type=float, default=0.008,
                   help="每帧暂停时间（秒），用于控制播放速度，默认 0.008")
    p.add_argument("--obs_mode", choices=("vision", "vision-touch"),
                   default="vision-touch",
                   help="观测模式，必须与训练时一致（默认 vision-touch）")
    p.add_argument("--perturb", action="store_true",
                   help="开启感知扰动（vision bias / force noise 等）")
    p.add_argument("--perturb_intensity", type=float, default=0.65)
    p.add_argument("--max_steps", type=int, default=220,
                   help="每个 episode 最大步数")
    p.add_argument("--stochastic", action="store_true",
                   help="使用随机策略（默认 deterministic）")
    return p.parse_args()


def load_policy(model_path: str):
    from stable_baselines3 import SAC
    print(f"[eval_viz] 加载模型: {model_path}")
    policy = SAC.load(model_path)
    print(f"[eval_viz] obs_dim={policy.observation_space.shape}, "
          f"act_dim={policy.action_space.shape}")
    return policy


def _run_one_episode(env: UR5DualPegEnv, policy, deterministic: bool, config,
                     viewer=None, sleep: float = 0.016):
    """运行一个 episode。viewer 为 None 时纯计算（无渲染）。"""
    obs = env.reset()
    success = False
    done_step = 0
    done_reason = "timeout"
    final_info = {}
    step_count = 0

    for _ in range(config.max_steps):
        if viewer is not None and not viewer.is_running():
            break

        if policy is not None:
            action, _ = policy.predict(obs, deterministic=deterministic)
        else:
            action = env.action_space.sample()

        obs, _, done, info = env.step(action)
        final_info = info
        step_count += 1

        if viewer is not None:
            viewer.sync()
            time.sleep(sleep)

        if done:
            success = bool(info.get("is_success", False))
            done_step = step_count
            done_reason = info.get("done_reason", "timeout")
            if viewer is not None:
                time.sleep(1.2)  # episode 结束后停顿，让用户看到结果
            break

    return success, done_step, done_reason, final_info


def main():
    args = parse_args()

    # 解析模型路径
    model_path = args.model
    if args.best and model_path is None:
        model_path = _DEFAULT_BEST_PATTERN
    if model_path is not None and not Path(model_path).exists():
        print(f"[警告] 模型文件不存在: {model_path}，将使用随机策略")
        model_path = None

    policy = load_policy(model_path) if model_path else None
    if policy is None:
        print("[eval_viz] 未加载模型，使用随机策略（成功率预计接近 0%）")

    # 构建 config
    updates = {}
    if args.obs_mode == "vision":
        updates["use_force_torque_obs"] = False
    if args.perturb:
        updates["perturb_enable"] = True
        updates["perturb_intensity"] = args.perturb_intensity
    updates["max_steps"] = args.max_steps
    config = replace(DualPegTaskConfig(), **updates)

    deterministic = not args.stochastic

    # ---------- 开始评估 ----------
    print(f"\n[eval_viz] episodes={args.episodes}  render={not args.no_render}"
          f"  deterministic={deterministic}  obs_mode={args.obs_mode}"
          f"  perturb={args.perturb}\n")

    results = []
    reason_counts: dict[str, int] = defaultdict(int)

    # 创建一个共享的 env（model 固定，data 在每次 reset 时原地更新）
    first_seed = args.seed_base
    env = UR5DualPegEnv(config=config, seed=first_seed)

    def _run_all(viewer=None):
        for ep in range(args.episodes):
            if viewer is not None and not viewer.is_running():
                break

            seed = args.seed_base + ep
            # 用不同 seed 重置随机状态
            env._np_random = np.random.default_rng(seed)

            success, done_step, done_reason, info = _run_one_episode(
                env, policy, deterministic, config,
                viewer=viewer,
                sleep=args.sleep,
            )

            reason_counts[done_reason] += 1
            rec = {
                "ep": ep,
                "seed": seed,
                "success": int(success),
                "done_step": done_step,
                "done_reason": done_reason,
                "worst_xy_err_mm": float(info.get("worst_xy_err", float("nan"))) * 1000,
                "worst_depth_sf_mm": float(info.get("worst_depth_shortfall", float("nan"))) * 1000,
            }
            results.append(rec)

            tag = "✓ 成功" if success else "✗ 失败"
            print(
                f"  ep={ep:3d}  seed={seed:4d}  {tag}  "
                f"step={done_step:3d}  reason={done_reason:<22s}  "
                f"xy={rec['worst_xy_err_mm']:5.2f}mm  "
                f"depth_sf={rec['worst_depth_sf_mm']:5.2f}mm"
            )

    if args.no_render:
        _run_all(viewer=None)
    else:
        # 单一持久 viewer 窗口，跑完所有 episode 再关闭
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            viewer.cam.distance = 0.65
            viewer.cam.lookat[:] = [0.0, 0.25, 0.93]
            viewer.cam.elevation = -18
            viewer.cam.azimuth = 90
            print("[eval_viz] MuJoCo 窗口已打开，开始逐 episode 播放...\n")
            _run_all(viewer=viewer)
            print("\n[eval_viz] 所有 episode 结束，窗口保持打开（手动关闭退出）...")
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.05)

    env.close()

    # ---------- 汇总 ----------
    n = len(results)
    if n == 0:
        return
    n_ok = sum(r["success"] for r in results)
    rate = n_ok / n
    success_steps = [r["done_step"] for r in results if r["success"]]

    print(f"\n{'='*60}")
    print(f"  成功率:  {n_ok}/{n} = {rate:.1%}")
    if success_steps:
        print(f"  成功平均步数: {np.mean(success_steps):.1f}  "
              f"(min={min(success_steps)}, max={max(success_steps)})")
    print("  失败/成功原因分布:")
    for reason, cnt in sorted(reason_counts.items()):
        bar = "█" * cnt
        print(f"    {reason:<24s} {cnt:3d} ({cnt/n:.1%})  {bar}")
    print("=" * 60)


if __name__ == "__main__":
    main()
