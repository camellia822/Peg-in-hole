from __future__ import annotations

import argparse
from collections import deque
from dataclasses import replace
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from pih_rebuild.config import DualPegTaskConfig
from pih_rebuild.envs.ur5_dual_peg_env import UR5DualPegEnv
from pih_rebuild.spar.algorithm import SPARSAC
from pih_rebuild.spar.buffers import SPARReplayBuffer
from pih_rebuild.spar.policies import SPARPolicy


INFO_KEYS = (
    "is_success",
    "episode_steps",
    "worst_dist",
    "worst_xy_err",
    "left_xy_err",
    "right_xy_err",
    "sync_xy_err",
    "e_xy_mean",
    "final_e_xy",
    "min_entry_depth",
    "left_entry_depth",
    "right_entry_depth",
    "left_depth_shortfall",
    "right_depth_shortfall",
    "worst_depth_shortfall",
    "depth_gap_mean",
    "final_depth_gap",
    "delta_depth_mean",
    "entry_progress",
    "xy_progress",
    "xy_ready_delta",
    "xy_ready_progress",
    "xy_excess",
    "terminal_depth_progress",
    "depth_closeness",
    "depth_score",
    "depth_sync_err",
    "near_xy_rate",
    "near_xy_steps",
    "stuck_rate",
    "done_reason",
    "done_reason_code",
    "failure_reason",
    "failure_reason_code",
    "insert_weight",
    "inserted_weight",
    "reward_align",
    "reward_xy_progress",
    "reward_xy_distance",
    "reward_insert",
    "reward_depth_distance",
    "reward_depth_level",
    "reward_depth_sync",
    "reward_terminal_depth",
    "reward_insert_xy_hold",
    "reward_force",
    "reward_prealign_press",
    "reward_aligned_z_action",
    "reward_depth_delta",
    "r_xy_progress",
    "r_xy_distance",
    "r_depth",
    "r_success",
    "r_time",
    "r_stuck",
    "r_action",
    "r_force",
    "depth_delta",
    "xy_far_down_weight",
    "action_xy_scale",
    "action_z_scale",
    "action_insert_weight",
    "action_depth_near_weight",
    "action_norm",
    "action_norm_near_xy",
    "force_norm",
    "torque_norm",
    "force_weight",
    "force_excess",
    "torque_excess",
    "vision_occluded",
    "vision_bias_norm",
    "force_bias_norm",
)


def _is_scalar_number(value) -> bool:
    if isinstance(value, (str, bytes)):
        return False
    if not np.isscalar(value):
        return False
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


class RebuildDiagnosticsCallback(BaseCallback):
    def __init__(
        self,
        window_size: int = 30,
        log_freq: int = 1000,
        best_model_path: Path | None = None,
        eval_freq: int = 0,
        eval_seeds: int = 0,
        best_eval_model_path: Path | None = None,
        env_config: DualPegTaskConfig | None = None,
    ):
        super().__init__()
        self.window_size = window_size
        self.log_freq = log_freq
        self.best_model_path = best_model_path
        self.eval_freq = eval_freq
        self.eval_seeds = eval_seeds
        self.best_eval_model_path = best_eval_model_path
        self.env_config = env_config
        self.episode_successes: deque[float] = deque(maxlen=window_size)
        self.episode_lengths: deque[int] = deque(maxlen=window_size)
        self.latest_info: dict = {}
        self.best_window_success_rate = -np.inf
        self.best_eval_success_rate = -np.inf
        self.best_eval_mean_done_step = np.inf

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for env_idx, info in enumerate(infos):
            if not info:
                continue
            self.latest_info = info
            done = bool(dones[env_idx]) if env_idx < len(dones) else False
            if done:
                self.episode_successes.append(float(info.get("is_success", 0.0)))
                episode = info.get("episode")
                if episode is not None:
                    self.episode_lengths.append(int(episode.get("l", 0)))

                if self.best_model_path is not None and len(self.episode_successes) == self.window_size:
                    window_success = float(np.mean(self.episode_successes))
                    if window_success > self.best_window_success_rate:
                        self.best_window_success_rate = window_success
                        self.model.save(self.best_model_path)

        if self.n_calls % self.log_freq == 0 and self.latest_info:
            if self.episode_successes:
                self.logger.record("rebuild/success_rate_window", float(np.mean(self.episode_successes)))
            if self.episode_lengths:
                self.logger.record("rebuild/episode_len_window", float(np.mean(self.episode_lengths)))
            for key in INFO_KEYS:
                value = self.latest_info.get(key)
                if _is_scalar_number(value):
                    self.logger.record(f"rebuild/{key}", float(value))
            self._print_console_metrics()
        if self.eval_freq > 0 and self.eval_seeds > 0 and self.n_calls % self.eval_freq == 0:
            self._run_deterministic_eval()
        return True

    def _print_console_metrics(self) -> None:
        step = self.n_calls
        info = self.latest_info
        success_rate = float(np.mean(self.episode_successes)) if self.episode_successes else float("nan")
        ep_len = float(np.mean(self.episode_lengths)) if self.episode_lengths else float("nan")
        worst_xy = info.get("worst_xy_err", float("nan"))
        worst_depth = info.get("worst_depth_shortfall", float("nan"))
        force_norm = info.get("force_norm", float("nan"))
        done_reason = info.get("done_reason", "")
        occ = info.get("vision_occluded", float("nan"))
        bias = info.get("vision_bias_norm", float("nan"))
        force_bias = info.get("force_bias_norm", float("nan"))
        print(
            f"[{step:>8d}] "
            f"success={success_rate:.3f}  ep_len={ep_len:>5.1f}  "
            f"xy_err={worst_xy*1000:>5.2f}mm  depth_sf={worst_depth*1000:>5.2f}mm  "
            f"F={force_norm:>5.1f}N  "
            f"occ={occ:.0f}  vis_bias={bias*1000:.2f}mm  F_bias={force_bias:.1f}N  "
            f"reason={done_reason}",
            flush=True,
        )

    def _run_deterministic_eval(self) -> None:
        rows = []
        reason_counts: dict[str, int] = {}
        for seed in range(self.eval_seeds):
            env = UR5DualPegEnv(config=self.env_config, seed=seed)
            obs = env.reset(seed=seed)
            final_info = {}
            success = 0.0
            done_step = env.config.max_steps - 1
            for step in range(env.config.max_steps):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, done, info = env.step(action)
                final_info = info
                if done:
                    success = float(info.get("is_success", 0.0))
                    done_step = step
                    reason = str(info.get("done_reason", "unknown"))
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
                    break
            rows.append(
                {
                    "success": success,
                    "done_step": float(done_step),
                    "final_worst_dist": float(final_info.get("worst_dist", np.nan)),
                    "final_e_xy": float(final_info.get("worst_xy_err", np.nan)),
                    "final_depth_gap": float(final_info.get("worst_depth_shortfall", np.nan)),
                }
            )
            env.close()

        success_rate = float(np.mean([row["success"] for row in rows]))
        mean_done_step = float(np.mean([row["done_step"] for row in rows]))
        mean_final_worst_dist = float(np.nanmean([row["final_worst_dist"] for row in rows]))
        mean_final_e_xy = float(np.nanmean([row["final_e_xy"] for row in rows]))
        mean_final_depth_gap = float(np.nanmean([row["final_depth_gap"] for row in rows]))
        depth_gap_rate = float(reason_counts.get("depth_gap", 0) / self.eval_seeds)
        stuck_rate = float(reason_counts.get("stuck_near_xy", 0) / self.eval_seeds)
        out_of_workspace_rate = float(reason_counts.get("out_of_workspace", 0) / self.eval_seeds)

        self.logger.record("eval_deterministic/success_rate", success_rate)
        self.logger.record("eval_deterministic/mean_done_step", mean_done_step)
        self.logger.record("eval_deterministic/mean_final_worst_dist", mean_final_worst_dist)
        self.logger.record("eval_deterministic/mean_final_e_xy", mean_final_e_xy)
        self.logger.record("eval_deterministic/mean_final_depth_gap", mean_final_depth_gap)
        self.logger.record("eval_deterministic/depth_gap_rate", depth_gap_rate)
        self.logger.record("eval_deterministic/stuck_near_xy_rate", stuck_rate)
        self.logger.record("eval_deterministic/out_of_workspace_rate", out_of_workspace_rate)
        self.logger.dump(self.num_timesteps)

        better_success = success_rate > self.best_eval_success_rate
        same_success_faster = success_rate == self.best_eval_success_rate and mean_done_step < self.best_eval_mean_done_step
        if self.best_eval_model_path is not None and (better_success or same_success_faster):
            self.best_eval_success_rate = success_rate
            self.best_eval_mean_done_step = mean_done_step
            self.model.save(self.best_eval_model_path)


def make_env(seed: int, monitor_file: Path, config: DualPegTaskConfig) -> Monitor:
    env = UR5DualPegEnv(config=config, seed=seed)
    return Monitor(env, filename=str(monitor_file), info_keywords=INFO_KEYS)


def build_config(args: argparse.Namespace) -> DualPegTaskConfig:
    config = DualPegTaskConfig()
    updates = {}
    if args.max_steps is not None:
        updates["max_steps"] = int(args.max_steps)
    if args.init_xy_random_mm is not None:
        init_xy_random = abs(float(args.init_xy_random_mm)) / 1000.0
        updates["init_xy_random"] = (-init_xy_random, init_xy_random)
    if args.init_height_mm is not None:
        updates["init_height_above_surface"] = abs(float(args.init_height_mm)) / 1000.0
    if args.workspace_xy_mm is not None:
        updates["workspace_xy_limit"] = abs(float(args.workspace_xy_mm)) / 1000.0
    if args.workspace_z_top_mm is not None:
        updates["workspace_z_top"] = abs(float(args.workspace_z_top_mm)) / 1000.0
    if args.oow_xy_mm is not None:
        updates["out_of_workspace_radius"] = abs(float(args.oow_xy_mm)) / 1000.0
    if args.goal_xy_random_mm is not None:
        goal_xy_random = abs(float(args.goal_xy_random_mm)) / 1000.0
        updates["goal_xy_random"] = (-goal_xy_random, goal_xy_random)
    if args.insertion_depth_mm is not None:
        insertion_depth = float(args.insertion_depth_mm) / 1000.0
        updates["insertion_depth"] = insertion_depth
        updates["depth_ref"] = insertion_depth
        updates["obs_depth_ref"] = insertion_depth
    if args.obs_mode == "vision":
        updates.update(
            {
                "use_force_torque_obs": False,
                "force_weight_preinsert": 0.0,
                "force_weight_inserted": 0.0,
            }
        )
    elif args.obs_mode == "vision-touch":
        updates["use_force_torque_obs"] = True
    else:
        raise ValueError(f"Unsupported obs_mode: {args.obs_mode}")
    if args.perturb:
        updates["perturb_enable"] = True
        updates["perturb_intensity"] = float(args.perturb_intensity)
    if args.vision_bias_xy_std is not None:
        updates["vision_bias_xy_std"] = float(args.vision_bias_xy_std)
    if args.vision_bias_z_std is not None:
        updates["vision_bias_z_std"] = float(args.vision_bias_z_std)
    if args.force_bias_drift_std is not None:
        updates["force_bias_drift_std"] = float(args.force_bias_drift_std)
    if args.force_bias_drift_max is not None:
        updates["force_bias_drift_max"] = float(args.force_bias_drift_max)
    if args.torque_bias_drift_std is not None:
        updates["torque_bias_drift_std"] = float(args.torque_bias_drift_std)
    if args.torque_bias_drift_max is not None:
        updates["torque_bias_drift_max"] = float(args.torque_bias_drift_max)
    if args.vision_noise_std is not None:
        updates["vision_noise_std"] = float(args.vision_noise_std)
    if args.force_noise_std is not None:
        updates["force_noise_std"] = float(args.force_noise_std)
    if args.torque_noise_std is not None:
        updates["torque_noise_std"] = float(args.torque_noise_std)
    if args.vision_occlusion_force is not None:
        updates["vision_occlusion_force"] = float(args.vision_occlusion_force)
    if args.vision_occlusion_depth is not None:
        updates["vision_occlusion_depth"] = float(args.vision_occlusion_depth)
    return replace(config, **updates)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAC on the rebuilt UR5 dual-peg task.")
    parser.add_argument("--timesteps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", type=str, default="rebuild_sac_v0")
    parser.add_argument("--log_freq", type=int, default=1000)
    parser.add_argument(
        "--success_window",
        type=int,
        default=30,
        help="Episode window for rebuild/success_rate_window and best_window_model selection.",
    )
    parser.add_argument(
        "--stats_window_size",
        type=int,
        default=100,
        help="SB3 rollout statistics window size for rollout/success_rate and ep_rew_mean.",
    )
    parser.add_argument(
        "--tb_log_interval",
        type=int,
        default=4,
        help="SB3 TensorBoard dump interval in episodes. Use 1 for denser curves.",
    )
    parser.add_argument("--learning_starts", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--ent_coef", type=str, default="auto_0.2")
    parser.add_argument(
        "--algo",
        choices=("sac", "m1", "m1m2", "m1m2m3"),
        default="sac",
        help="Which SPAR-SAC ablation to train: sac | m1 | m1m2 | m1m2m3.",
    )
    parser.add_argument("--m1_warmup", type=int, default=20000, help="Steps before the M1 phase gate starts ramping.")
    parser.add_argument("--m1_ramp", type=int, default=30000, help="Ramp length (steps) for the M1 phase gate 0->1.")
    parser.add_argument("--m2_warmup", type=int, default=20000, help="Steps before the M2 dynamic-entropy blend starts.")
    parser.add_argument("--m2_ramp", type=int, default=30000, help="Ramp length (steps) for the M2 blend 0->1.")
    parser.add_argument("--phase_enc_dim", type=int, default=16, help="Width of the phase-conditioning encoder.")
    parser.add_argument("--phase_hidden", type=int, default=64, help="Hidden width of the PhaseNet trunk.")
    parser.add_argument("--eval_freq", type=int, default=25000)
    parser.add_argument("--eval_seeds", type=int, default=20)
    parser.add_argument("--obs_mode", choices=("vision", "vision-touch"), default="vision-touch")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument(
        "--init_xy_random_mm",
        type=float,
        default=None,
        help="Override symmetric initial XY randomization range in millimeters (e.g. 80 -> [-80mm, 80mm]).",
    )
    parser.add_argument(
        "--init_height_mm",
        type=float,
        default=None,
        help="Override initial EE height above the hole surface in millimeters (default 80mm).",
    )
    parser.add_argument(
        "--workspace_xy_mm",
        type=float,
        default=None,
        help="Override EE-target XY workspace half-extent in millimeters (default 80mm). Enlarge so init_xy_random stays effective.",
    )
    parser.add_argument(
        "--workspace_z_top_mm",
        type=float,
        default=None,
        help="Override EE-target max height above the hole surface in millimeters (default 140mm). Enlarge so init_height stays effective.",
    )
    parser.add_argument(
        "--oow_xy_mm",
        type=float,
        default=None,
        help="Override out-of-workspace XY radius in millimeters (default 120mm). Enlarge together with the workspace bounds.",
    )
    parser.add_argument(
        "--goal_xy_random_mm",
        type=float,
        default=None,
        help="Override symmetric hole XY randomization range in millimeters (e.g. 2 -> [-2mm, 2mm]).",
    )
    parser.add_argument(
        "--insertion_depth_mm",
        type=float,
        default=None,
        help="Override target insertion depth in millimeters (also updates depth_ref and obs_depth_ref).",
    )
    parser.add_argument(
        "--perturb",
        action="store_true",
        help="Enable observation-only perception perturbations (vision bias, contact occlusion, force noise).",
    )
    parser.add_argument(
        "--perturb_intensity",
        type=float,
        default=1.0,
        help="Scalar that scales every perturbation channel (e.g. 0.5..2.0). Only used with --perturb.",
    )
    parser.add_argument(
        "--vision_bias_xy_std",
        type=float,
        default=None,
        help="Override vision static XY bias std (meters) for single-factor ablations.",
    )
    parser.add_argument(
        "--vision_bias_z_std",
        type=float,
        default=None,
        help="Override vision static Z bias std (meters) for single-factor ablations.",
    )
    parser.add_argument(
        "--force_bias_drift_std",
        type=float,
        default=None,
        help="Override force bias random-walk std (N/step) for single-factor drift ablations.",
    )
    parser.add_argument(
        "--force_bias_drift_max",
        type=float,
        default=None,
        help="Override force bias random-walk clamp (N).",
    )
    parser.add_argument(
        "--torque_bias_drift_std",
        type=float,
        default=None,
        help="Override torque bias random-walk std (N*m/step) for single-factor drift ablations.",
    )
    parser.add_argument(
        "--torque_bias_drift_max",
        type=float,
        default=None,
        help="Override torque bias random-walk clamp (N*m).",
    )
    parser.add_argument(
        "--vision_noise_std",
        type=float,
        default=None,
        help="Override per-step Gaussian vision noise std (meters) for single-factor ablations.",
    )
    parser.add_argument(
        "--force_noise_std",
        type=float,
        default=None,
        help="Override force sensor per-step Gaussian noise std (N) for single-factor ablations.",
    )
    parser.add_argument(
        "--torque_noise_std",
        type=float,
        default=None,
        help="Override torque sensor per-step Gaussian noise std (N*m) for single-factor ablations.",
    )
    parser.add_argument(
        "--vision_occlusion_force",
        type=float,
        default=None,
        help="Override vision occlusion force threshold (N). Vision is frozen when ||F|| >= this value.",
    )
    parser.add_argument(
        "--vision_occlusion_depth",
        type=float,
        default=None,
        help="Override vision occlusion depth threshold (m). Vision is frozen when insertion depth >= this value.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_config = build_config(args)
    step_tag = f"h{env_config.max_steps}"
    perturb_tag = f"_perturb{args.perturb_intensity:g}" if args.perturb else ""
    run_name = f"{args.tag}_{args.obs_mode}_{step_tag}{perturb_tag}_steps{args.timesteps}_seed{args.seed}"
    output_dir = Path("output") / "rebuild" / run_name
    model_dir = output_dir / "model"
    tensorboard_dir = output_dir / "tensorboard"
    monitor_dir = output_dir / "monitor"
    model_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    monitor_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(args.seed, monitor_dir / "train_monitor.csv", env_config)
    common_kwargs = dict(
        seed=args.seed,
        verbose=1,
        tensorboard_log=str(tensorboard_dir),
        learning_rate=args.learning_rate,
        buffer_size=100_000,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        gamma=0.98,
        train_freq=1,
        gradient_steps=1,
        ent_coef=args.ent_coef if args.ent_coef.startswith("auto") else float(args.ent_coef),
        stats_window_size=args.stats_window_size,
    )
    algo = args.algo.lower()
    if algo == "sac":
        model = SAC("MlpPolicy", env, **common_kwargs)
        tb_log_name = "SAC"
    else:
        enable_m2 = algo in ("m1m2", "m1m2m3")
        enable_m3 = algo == "m1m2m3"
        policy_kwargs = dict(
            phase_enc_dim=args.phase_enc_dim,
            phase_hidden=args.phase_hidden,
            n_phases=4,
        )
        model = SPARSAC(
            SPARPolicy,
            env,
            enable_m1=True,
            enable_m2=enable_m2,
            enable_m3=enable_m3,
            m1_warmup=args.m1_warmup,
            m1_ramp=args.m1_ramp,
            m2_warmup=args.m2_warmup,
            m2_ramp=args.m2_ramp,
            policy_kwargs=policy_kwargs,
            replay_buffer_class=SPARReplayBuffer,
            **common_kwargs,
        )
        tb_log_name = algo.upper()
    callback = RebuildDiagnosticsCallback(
        window_size=args.success_window,
        log_freq=args.log_freq,
        best_model_path=model_dir / "best_window_model",
        eval_freq=args.eval_freq,
        eval_seeds=args.eval_seeds,
        best_eval_model_path=model_dir / "best_eval_model",
        env_config=env_config,
    )
    model.learn(
        total_timesteps=args.timesteps,
        callback=callback,
        tb_log_name=tb_log_name,
        log_interval=args.tb_log_interval,
    )
    model.save(model_dir / "final_model")
    env.close()

    if callback.episode_successes:
        success_rate = float(np.mean(callback.episode_successes))
        print(f"final_window_success_rate={success_rate:.3f}")
    if np.isfinite(callback.best_window_success_rate):
        print(f"best_window_success_rate={callback.best_window_success_rate:.3f}")
        print(f"saved_best_model={model_dir / 'best_window_model.zip'}")
    if np.isfinite(callback.best_eval_success_rate):
        print(f"best_eval_success_rate={callback.best_eval_success_rate:.3f}")
        print(f"best_eval_mean_done_step={callback.best_eval_mean_done_step:.1f}")
        print(f"saved_best_eval_model={model_dir / 'best_eval_model.zip'}")
    print(f"saved_model={model_dir / 'final_model.zip'}")
    print(f"tensorboard_log={tensorboard_dir}")
    print(f"monitor_csv={monitor_dir / 'train_monitor.csv'}")


if __name__ == "__main__":
    main()
