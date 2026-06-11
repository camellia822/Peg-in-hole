# Reward V1 Strong Snapshot

This file records reward version `v1`: the strong-shaped pure-vision baseline
before weakening the reward for a harder SAC baseline.

## Task

- XML: `pih_rebuild/assets/mujoco/ur5_dual_peg.xml`
- Robot: UR5 with dual cylindrical pegs.
- Hole/peg clearance: single-side `1.0 mm`
- Peg radius: `0.0200 m`
- Radial clearance: `0.0010 m`
- Insertion depth: `0.020 m`
- Success XY threshold: `0.0010 m`
- Success depth tolerance: `0.00015 m`
- Initial XY random range at snapshot time: `[-0.050, 0.050] m`
- Initial height above hole surface: `0.080 m`
- Typical training override: `--max_steps 200`

## Pure Vision Observation

The current `vision` mode is not an RGB/depth image. It is a privileged
geometry/state vector built from MuJoCo site positions.

Pure vision observation dimension: `18`

- `left_err / obs_error_ref`: 3 values
- `right_err / obs_error_ref`: 3 values
- `left_xy_err / obs_xy_ref`
- `right_xy_err / obs_xy_ref`
- `worst_xy_err / obs_xy_ref`
- `sync_xy_err / obs_xy_ref`
- `left_depth_shortfall / obs_depth_ref`
- `right_depth_shortfall / obs_depth_ref`
- `worst_depth_shortfall / obs_depth_ref`
- `min_entry_depth / obs_depth_ref`
- `depth_sync_err / obs_depth_ref`
- `insert_weight`
- `inserted_weight`
- `step_count / max_steps`

Normalization refs:

- `obs_error_ref = 0.03`
- `obs_xy_ref = 0.015`
- `obs_depth_ref = 0.020`
- observation clipping: `[-10, 10]`

In `--obs_mode vision`, force/torque observations are disabled and force reward
weights are set to zero in `build_config`.

## Action

Policy output: `action = [ax, ay, az]`, each in `[-1, 1]`.

The environment maps the action to task-space delta xyz, then uses IK to
produce UR5 joint targets.

Stage weights:

- `insert_weight = sigmoid((align_xy_gate - worst_xy_err) / align_xy_tau)`
- `depth_near_weight = sigmoid((depth_action_near_gate - worst_depth_shortfall) / depth_action_tau)`

Strong-version action scales:

- `xy_action_scale_far = 0.0018`
- `xy_action_scale_approach = 0.00045`
- `xy_action_scale_insert = 0.00012`
- `xy_action_far_gate = 0.018`
- `z_action_scale_far = 0.00055`
- `z_action_scale_insert = 0.00016`
- `z_action_scale_prealign = 0.00025`
- `depth_action_near_gate = 0.003`
- `depth_action_tau = 0.0008`
- `align_xy_gate = 0.0010`
- `align_xy_tau = 0.00030`

## Strong Reward Formula

Total reward:

```text
reward =
    r_xy_progress
  + r_xy_distance
  + r_insert
  + r_terminal_depth
  + r_insert_xy_hold
  + r_prealign_press
  + r_aligned_z_action
  + r_depth_delta
  + r_stuck
  + r_force
  + r_action
  + r_time
  + r_success
```

For pure vision, `r_force = 0` because force weights are disabled by
`--obs_mode vision`.

Strong shaping weights:

- `success_bonus = 50.0`
- `xy_progress_weight = 0.70`
- `xy_ready_bonus_weight = 0.35`
- `depth_distance_weight = 0.55`
- `depth_level_weight = 0.35`
- `terminal_depth_weight = 0.45`
- `insert_xy_hold_weight = 0.45`
- `prealign_press_weight = 0.08`
- `aligned_down_action_weight = 0.32`
- `aligned_up_action_weight = 0.12`
- `depth_delta_reward_weight = 1.00`
- `stuck_penalty_weight = 0.06`
- `action_penalty_weight = 0.010`
- `time_penalty = 0.010`

The strongest teacher-like terms are:

- `xy_ready_bonus_weight`: rewards entering the insertion gate.
- `aligned_down_action_weight`: directly rewards downward action after XY alignment.
- `depth_delta_reward_weight`: directly rewards immediate depth progress.
- `prealign_press_weight`: directly discourages downward action before alignment.

These terms made SAC learn very quickly and left little room for method
improvement.
