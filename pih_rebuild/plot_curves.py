#!/usr/bin/env python3
"""论文风格训练曲线绘图脚本。

从 Monitor CSV（每 episode 一行）绘制 成功率/奖励 vs 训练步数 曲线，
风格对齐常见 RL 论文（平滑主线 + 浅色原始曲线 / 多种子均值±std 带）。

用法示例:
  # 单条曲线
  python -m pih_rebuild.plot_curves \
      --run "SAC=output/rebuild/sac_p065_seed7_vision-touch_h220_perturb0.65_steps200000_seed7" \
      --out output/analysis/p065

  # 多算法对比（每个 --run 一条曲线）
  python -m pih_rebuild.plot_curves \
      --run "SAC=output/rebuild/<sac_run>" \
      --run "SAC+M1=output/rebuild/<m1_run>" \
      --run "SAC+M1+M2=output/rebuild/<m1m2_run>" \
      --out output/analysis/compare

  # 多种子：同一 label 用逗号列出多个 run 目录，画均值±std 带
  python -m pih_rebuild.plot_curves \
      --run "SAC=run_seed7,run_seed17,run_seed27" \
      --out output/analysis/sac_3seeds
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
    from tensorboard.compat.proto.event_pb2 import Event
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

# 论文常见配色（蓝、绿、红、橙、紫、青）
COLORS = ["#1f77b4", "#2ca02c", "#d62728", "#ff7f0e", "#9467bd", "#17becf"]


def parse_args():
    p = argparse.ArgumentParser(description="论文风格训练曲线")
    p.add_argument("--run", action="append", required=True,
                   help="格式 'label=run_dir' 或 'label=dir1,dir2,...'（多种子）。"
                        "run_dir 可以是实验根目录（自动找 monitor/train_monitor.csv）"
                        "或直接是 csv 路径")
    p.add_argument("--out", type=str, required=True,
                   help="输出前缀，生成 <out>_success.png 和 <out>_reward.png")
    p.add_argument("--window", type=int, default=50,
                   help="滑动窗口大小（episode 数），默认 50")
    p.add_argument("--max_steps", type=float, default=None,
                   help="x 轴截断（步数），默认画全程")
    p.add_argument("--grid_points", type=int, default=300,
                   help="多种子插值网格点数")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--add-eval", action="store_true",
                   help="添加确定性评估曲线（虚线）")
    return p.parse_args()


def resolve_csv(path_str: str) -> Path:
    path = Path(path_str)
    if path.suffix == ".csv":
        return path
    cand = path / "monitor" / "train_monitor.csv"
    if cand.exists():
        return cand
    cand2 = path / "train_monitor.csv"
    if cand2.exists():
        return cand2
    raise FileNotFoundError(f"找不到 monitor csv: {path_str}")

def load_tb_eval_data(run_dir: str, metric: str = "eval_deterministic/success_rate"):
    """从 TensorBoard 事件日志读取评估数据。返回 (steps, values)，不存在时返回 (None, None)"""
    if not TENSORBOARD_AVAILABLE:
        return None, None
    
    tb_dir = Path(run_dir) / "tensorboard"
    if not tb_dir.exists():
        return None, None
    
    steps_list = []
    values_list = []
    
    # 遍历 tensorboard 目录下的所有事件文件（包括子目录）
    try:
        event_files = list(tb_dir.rglob("events.out.tfevents.*"))
        # print(f"[debug] Found {len(event_files)} event files in {tb_dir}")
        for event_file in sorted(event_files):
            loader = EventFileLoader(str(event_file))
            for event in loader.Load():
                if event.HasField("summary"):
                    for value in event.summary.value:
                        if value.tag == metric and value.HasField("simple_value"):
                            steps_list.append(event.step)
                            values_list.append(value.simple_value)
    except Exception as e:
        print(f"[warn] 读取 TensorBoard 数据失败: {e}")
        return None, None
    
    if not steps_list:
        return None, None
    
    return np.array(steps_list, dtype=np.float64), np.array(values_list, dtype=np.float64)

        print(f"[debug] Found {len(event_files)} event files in {tb_dir}")
        for event_file in sorted(event_files):
            loader = EventFileLoader(str(event_file))
    print(f"[debug] Found {len(event_files)} event files in {tb_dir}")
def load_run(csv_path: Path, window: int):
    """返回 (steps, success_smooth, reward_smooth)。"""
    df = pd.read_csv(csv_path, skiprows=1)
    steps = df["l"].cumsum().to_numpy(dtype=np.float64)
    def _to01(v):
        if isinstance(v, (bool, np.bool_)):
            return float(v)
        s = str(v).strip().lower()
        if s in ("true", "1", "1.0"):
            return 1.0
        if s in ("false", "0", "0.0"):
            return 0.0
        return float(v)

    succ = df["is_success"].map(_to01).to_numpy(dtype=np.float64)
    rew = df["r"].to_numpy(dtype=np.float64)

    w = max(2, min(window, len(df) // 2))
    succ_s = pd.Series(succ).rolling(w, min_periods=1).mean().to_numpy()
    rew_s = pd.Series(rew).rolling(w, min_periods=1).mean().to_numpy()
    return steps, succ, succ_s, rew, rew_s


def interp_to_grid(runs, grid_points, value_idx):
    """把多个 run 的曲线插值到公共步数网格。value_idx: 2=success_smooth, 4=reward_smooth"""
    max_step = min(r[0][-1] for r in runs)
    grid = np.linspace(0, max_step, grid_points)
    mat = np.stack([
        np.interp(grid, r[0], r[value_idx]) for r in runs
    ])
    return grid, mat.mean(axis=0), mat.std(axis=0)


def style_axes(ax, xlabel, ylabel):
    ax.grid(True, alpha=0.3, linewidth=0.6)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)


def main():
    args = parse_args()
    out_prefix = Path(args.out)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    # 解析 runs
    entries = []  # (label, [run_tuples...])
    for spec in args.run:
        if "=" not in spec:
            raise ValueError(f"--run 需要 'label=path' 格式: {spec}")
        label, paths = spec.rsplit("=", 1)
        runs = []
        for ps in paths.split(","):
            csv = resolve_csv(ps.strip())
            runs.append(load_run(csv, args.window))
            print(f"[plot] {label}: {csv}  episodes={len(runs[-1][0])}  "
                  f"total_steps={runs[-1][0][-1]:.0f}")
        entries.append((label, runs))

    # 如果 --add-eval，从每个 run_dir 读取评估曲线
    eval_data = {}  # label -> [(run_dir, steps, values), ...]
    if args.add_eval:
        for spec in args.run:
            if "=" not in spec:
                continue
            label, paths = spec.rsplit("=", 1)
            eval_list = []
            for ps in paths.split(","):
                run_dir = ps.strip()
                tb_steps, tb_vals = load_tb_eval_data(run_dir)
                if tb_steps is not None:
                    eval_list.append((run_dir, tb_steps, tb_vals))
                    print(f"[eval] {label}: loaded eval from {run_dir}, {len(tb_steps)} points")
                else:
                    print(f"[eval] {label}: no eval data from {run_dir}")
            if eval_list:
                eval_data[label] = eval_list

    total = max(r[0][-1] for _, runs in entries for r in runs)
    if args.max_steps:
        total = min(total, args.max_steps)
    if total >= 1e6:
        scale, xlabel = 1e6, "Timesteps (×10⁶)"
    else:
        scale, xlabel = 1e3, "Timesteps (×10³)"

    # ---------- 成功率图 ----------
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for i, (label, runs) in enumerate(entries):
        c = COLORS[i % len(COLORS)]
        if len(runs) == 1:
            steps, _, smooth = runs[0][0], runs[0][1], runs[0][2]
            # 成功率原始值是 0/1，画出来是竖条，所以用短窗口曲线做浅色背景
            w_short = max(2, args.window // 4)
            raw_s = pd.Series(runs[0][1]).rolling(
                w_short, min_periods=1).mean().to_numpy()
            if args.max_steps:
                m = steps <= args.max_steps
                steps, raw_s, smooth = steps[m], raw_s[m], smooth[m]
            ax.plot(steps / scale, raw_s, color=c, alpha=0.22, lw=0.8)
            ax.plot(steps / scale, smooth, color=c, lw=1.8, label=label)
            # 添加评估曲线（虚线）
            if args.add_eval and label in eval_data and len(eval_data[label]) == 1:
                run_dir, tb_steps, tb_vals = eval_data[label][0]
                if args.max_steps:
                    m = tb_steps <= args.max_steps
                    tb_steps, tb_vals = tb_steps[m], tb_vals[m]
                ax.plot(tb_steps / scale, tb_vals, color=c, lw=1.5, linestyle="--", 
                        alpha=0.7, label=f"{label} (eval)")
        else:
            grid, mean, std = interp_to_grid(runs, args.grid_points, 2)
            if args.max_steps:
                m = grid <= args.max_steps
                grid, mean, std = grid[m], mean[m], std[m]
            ax.fill_between(grid / scale, mean - std, mean + std,
                            color=c, alpha=0.18, lw=0)
            ax.plot(grid / scale, mean, color=c, lw=1.8, label=label)
    ax.set_ylim(-0.02, 1.02)
    style_axes(ax, xlabel, "Success Rate")
    ax.legend(fontsize=10, loc="best", frameon=True)
    fig.tight_layout()
    f1 = f"{out_prefix}_success.png"
    fig.savefig(f1, dpi=args.dpi)
    plt.close(fig)
    print(f"[plot] saved {f1}")

    # ---------- 奖励图 ----------
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for i, (label, runs) in enumerate(entries):
        c = COLORS[i % len(COLORS)]
        if len(runs) == 1:
            steps, raw, smooth = runs[0][0], runs[0][3], runs[0][4]
            if args.max_steps:
                m = steps <= args.max_steps
                steps, raw, smooth = steps[m], raw[m], smooth[m]
            ax.plot(steps / scale, raw, color=c, alpha=0.18, lw=0.8)
            ax.plot(steps / scale, smooth, color=c, lw=1.8, label=label)
        else:
            grid, mean, std = interp_to_grid(runs, args.grid_points, 4)
            if args.max_steps:
                m = grid <= args.max_steps
                grid, mean, std = grid[m], mean[m], std[m]
            ax.fill_between(grid / scale, mean - std, mean + std,
                            color=c, alpha=0.18, lw=0)
            ax.plot(grid / scale, mean, color=c, lw=1.8, label=label)
    style_axes(ax, xlabel, "Episode Reward")
    ax.legend(fontsize=10, loc="best", frameon=True)
    fig.tight_layout()
    f2 = f"{out_prefix}_reward.png"
    fig.savefig(f2, dpi=args.dpi)
    plt.close(fig)
    print(f"[plot] saved {f2}")


if __name__ == "__main__":
    main()
