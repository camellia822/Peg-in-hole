# Reward Version Index

This file fixes the reward naming used for the rebuilt dual cylindrical peg
insertion experiments. Future reward designs should continue in order:
`v3`, `v4`, ...

## v1: Strong Shaping

Snapshot file: `pih_rebuild/reward_v1_strong_snapshot.md`

This is the original strong-shaped reward. It uses the same 18D pure-vision
geometry observation and the same adaptive action scales, but has stronger
teacher-like shaping terms:

- `success_bonus = 50.0`
- `xy_progress_weight = 0.70`
- `xy_ready_bonus_weight = 0.35`
- `depth_distance_weight = 0.55`
- `depth_level_weight = 0.35`
- `terminal_depth_weight = 0.45`
- `prealign_press_weight = 0.08`
- `aligned_down_action_weight = 0.32`
- `aligned_up_action_weight = 0.12`
- `depth_delta_reward_weight = 1.00`

## v2: Medium Shaping

Current code file: `pih_rebuild/config.py`

This is the current reward in code. It keeps the same observation design,
success condition, task geometry, and adaptive action scales, but weakens the
main shaping terms to make the SAC baseline less over-guided:

- `success_bonus = 35.0`
- `xy_progress_weight = 0.35`
- `xy_ready_bonus_weight = 0.05`
- `depth_distance_weight = 0.40`
- `depth_level_weight = 0.25`
- `terminal_depth_weight = 0.25`
- `prealign_press_weight = 0.02`
- `aligned_down_action_weight = 0.06`
- `aligned_up_action_weight = 0.04`
- `depth_delta_reward_weight = 0.20`

## Pure Vision Rule

For `--obs_mode vision`, the code uses the same reward formula but disables
force/torque observations and force penalty weights:

- `use_force_torque_obs = False`
- `force_weight_preinsert = 0.0`
- `force_weight_inserted = 0.0`

So the pure-vision reward version is:

- `v1-vision`: strong shaping with force terms disabled.
- `v2-vision`: current medium shaping with force terms disabled.

## v3: Force-Aware Medium Shaping

Current code file: `pih_rebuild/config.py` and
`pih_rebuild/envs/ur5_dual_peg_env.py`

This version keeps the v2 geometry reward and action-scale structure, then
adds a safer force/tactile path for `--obs_mode vision-touch`:

- Force/torque sensor values are zeroed at reset after the robot settles.
- The policy observes compensated force xyz, compensated torque xyz, force
  norm, torque norm, and a smooth contact gate.
- Force penalty is weak before insertion alignment and gradually becomes
  stronger as `insert_weight` / `inserted_weight` increases.
- Logged diagnostics include `force_weight`, `force_excess`, and
  `torque_excess`.

The intended naming is:

- `v3-vision-touch`: force-aware medium shaping with 27D observation.

## v4: Lean Dense (current default)

Current code file: `pih_rebuild/config.py` and
`pih_rebuild/envs/ur5_dual_peg_env.py`. This is now the reward in code; it
replaces v3 in place (no flag).

Motivation: v1/v2/v3 stacked many prescriptive shaping terms (press-down /
no-press-down action rewards, anti-stuck, xy-hold, depth-sync, terminal-depth,
xy-ready bonus). Those competing terms made the success curve rise slowly and
oscillate. v4 rewards task OUTCOMES and lets SAC discover the strategy:

- XY alignment: `align_progress_weight * clip(progress)` (clipped potential
  progress, signal at every scale, zero for hovering) + a small
  `align_closeness_weight * (1 - tanh(worst_xy_err / align_xy_ref))` gradient.
- Depth insertion: gated by `insert_weight` (only counts once aligned),
  `depth_progress_weight * clip(depth_delta)` + `depth_closeness_weight *
  depth_closeness`. Reward comes from depth actually increasing / being deep,
  never from a prescribed "press down" action.
- Sparse `success_bonus` (35) = the dominant incentive.
- Minimal regularizers only: contact-weighted force safety, `||a||^2` action
  smoothness, and a small time penalty.

Removed vs v3: `r_prealign_press`, `r_aligned_z_action`, `r_depth_delta` (old
form), `r_stuck`, `r_insert_xy_hold`, `r_depth_sync`, `r_terminal_depth`,
`r_depth_level`, and the `xy_ready` bonus / `xy_stage_weight` micro-tuning. The
info/CSV schema is unchanged: removed components log as `0.0` so old monitor
CSVs and plots stay comparable.

Action scales are unchanged (`_action_scales`). The clean baseline runs with no
perception perturbation (`experiments/run_sac_baseline_clean_seed7_200k.sh`):
no vision bias / noise / occlusion / force drift.

The intended naming is:

- `v4-vision-touch`: lean dense reward with 27D observation.
- `v4-vision`: same formula with force terms / force obs disabled
  (`--obs_mode vision`).
