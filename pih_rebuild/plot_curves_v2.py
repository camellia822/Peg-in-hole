#!/usr/bin/env python3
"""论文风格训练曲线绘图脚本（支持评估曲线）。

从 Monitor CSV（每 episode 一行）和 TensorBoard 评估日志绘制曲线。

用法示例:
  # 不含评估曲线
  python -m pih_rebuild.plot_curves_v2 \\
      --run "SAC=output/rebuild/sac_p065_seed7_vision-touch_h220_perturb0.65_steps200000_seed7" \\
      --out output/analysis/p065

  # 含评估曲线（虚线）
  python -m pih_rebuild.plot_curves_v2 \\
      --run "SAC=output/rebuild/sac_p065_seed7_vision-touch_h220_perturb0.65_steps200000_seed7" \\
      --out output/analysis/p065 \\
      --add-eval
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
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

COLORS = ["#1f77b4", "#2ca02c", "#d62728", "#ff7f0e", "#9467bd", "#17becf"]


def parse_args():
    p = argparse.ArgumentParser(description="论文风格训练曲线（支持评估曲线）")
    p.add_argument("--run", action="append", required=True,
                   help="格式 'label=run_dir' 或 'label=dir1,dir2,...'（多种子）")
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


def load_tb_eval(run_dir: str, metric: str = "eval_deterministic/success_rate"):
    """从 TensorBoard 读取评估曲线。返回 (steps, values) 或 (None, None)"""
    if not TENSORBOARD_AVAILABLE:
        return None, None
    
    tb_dir = Path(run_dir) / "tensorboard"
    if not tb_dir.exists():
        return None, None
    
    steps_list = []
    values_list = []
    
    try:
        # 递归查找所有事件文件
        for event_file in sorted(tb_dir.rglob("events.out.tfevents.*")):
            loader = EventFileLoader(str(event_file))
            for event in loader.Load():
                if event.HasField("summary"):
                    for value in event.summary.value:
                        if value.tag == metric and value.HasField("simple_value"):
                            steps_list.append(event.step)
                            values_list.append(value.simple_value)
    except Exception as e:
        print(f"[warn] 读取TensorBoard失败: {e}")
        return None, None
    
    if not steps_list:
        return None, None
    
    # 排序并去重
    data = sorted(zip(steps_list, values_list))
    steps = np.array([x[0] for x in data], dtype=np.float64)
    values = np.array([x[1] for x in data], dtype=np.float64)
    return steps, values


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
    run_dirs = {}  # label -> [run_dir, ...]（用于评估曲线）
    
    for spec in args.run:
        if "=" not in spec:
            raise ValueError(f"--run 需要 'label=path' 格式: {spec}")
        label, paths = spec.rsplit("=", 1)
        runs = []
        dirs = []
        for ps in paths.split(","):
            ps = ps.strip()
            csv = resolve_csv(ps)
            runs.append(load_run(csv, args.window))
            dirs.append(ps)
            print(f"[plot] {label}: {csv}  episodes={len(runs[-1][0])}  "
                  f"total_steps={runs[-1][0][-1]:.0f}")
        entries.append((label, runs))
        run_dirs[label] = dirs

    total = max(r[0][-1] for _, runs in entries for r in runs)
    if args.max_steps:
        total = min(total, args.max_steps)
    if total >= 1e6:
        scale, xlabel = 1e6, "Timesteps (×10⁶)"
    else:
        scale, xlabel = 1e3, "Timesteps (×10³)"

    # 加载评估曲线
    eval_curves = {}  # label -> [(steps, values), ...]
    if args.add_eval:
        for label, dirs in run_dirs.items():
            curves = []
            for d in dirs:
                steps, vals = load_tb_eval(d)
                if steps is not None:
                    curves.append((steps, vals))
                    print(f"[eval] {label}: loaded {len(steps)} eval points from {d}")
                else:
                    print(f"[eval] {label}: no eval data from {d}")
            if curves:
                eval_curves[label] = curves

    # ---------- 成功率图 ----------
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for i, (label, runs) in enumerate(entries):
        c = COLORS[i % len(COLORS)]
        if len(runs) == 1:
            steps, _, smooth = runs[0][0], runs[0][1], runs[0][2]
            w_short = max(2, args.window // 4)
            raw_s = pd.Series(runs[0][1]).rolling(
                w_short, min_periods=1).mean().to_numpy()
            if args.max_steps:
                m = steps <= args.max_steps
                steps, raw_s, smooth = steps[m], raw_s[m], smooth[m]
            ax.plot(steps / scale, raw_s, color=c, alpha=0.22, lw=0.8)
            ax.plot(steps / scale, smooth, color=c, lw=1.8, label=label)
            
            # 添加评估曲线
            if label in eval_curves and len(eval_curves[label]) == 1:
                eval_steps, eval_vals = eval_curves[label][0]
                if args.max_steps:
                    m = eval_steps <= args.max_steps
                    eval_steps, eval_vals = eval_steps[m], eval_vals[m]
                ax.plot(eval_steps / scale, eval_vals, color=c, lw=1.5, linestyle="--",
                        alpha=0.8, label=f"{label} (eval)")
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
