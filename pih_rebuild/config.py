from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DualPegTaskConfig:
    model_xml: Path = ROOT_DIR / "pih_rebuild" / "assets" / "mujoco" / "ur5_dual_peg.xml"
    urdf: Path = ROOT_DIR / "pih_rebuild" / "assets" / "urdf" / "ur5_robot.urdf"
    hole_surface_center: tuple[float, float, float] = (-0.05, 0.32, 0.88)
    goal_xy_random: tuple[float, float] = (0.0, 0.0)
    init_xy_random: tuple[float, float] = (-0.050, 0.050)
    init_height_above_surface: float = 0.080
    peg_radius: float = 0.0200
    radial_clearance: float = 0.0010
    insertion_depth: float = 0.020
    success_xy_threshold: float = 0.0010
    success_depth_tolerance: float = 0.00015
    success_threshold: float = 0.0010
    peg_spacing: float = 0.12
    xy_action_scale_far: float = 0.0018
    xy_action_scale_approach: float = 0.00045
    xy_action_scale_insert: float = 0.00012
    xy_action_far_gate: float = 0.018
    z_action_scale_far: float = 0.00055
    z_action_scale_insert: float = 0.00016
    z_action_scale_prealign: float = 0.00025
    depth_action_near_gate: float = 0.003
    depth_action_tau: float = 0.0008
    max_steps: int = 220
    control_substeps: int = 20
    obs_error_ref: float = 0.03
    obs_dist_ref: float = 0.03
    obs_xy_ref: float = 0.015
    obs_depth_ref: float = 0.020
    use_force_torque_obs: bool = True
    force_obs_ref: float = 80.0
    torque_obs_ref: float = 8.0
    force_norm_obs_ref: float = 120.0
    torque_norm_obs_ref: float = 12.0
    reward_dist_ref: float = 0.015
    success_bonus: float = 35.0
    align_xy_gate: float = 0.0010
    align_xy_tau: float = 0.00030
    align_xy_ref: float = 0.003
    align_sync_ref: float = 0.0005
    xy_progress_ref: float = 0.0006
    xy_progress_weight: float = 0.35
    xy_ready_bonus_weight: float = 0.05
    depth_ref: float = 0.020
    depth_sync_ref: float = 0.0007
    depth_distance_weight: float = 0.40
    depth_level_weight: float = 0.25
    inserted_gate: float = 0.0006
    inserted_tau: float = 0.0002
    force_safe: float = 45.0
    torque_safe: float = 4.0
    force_penalty_ref: float = 120.0
    torque_penalty_ref: float = 12.0
    force_weight_preinsert: float = 0.04
    force_weight_inserted: float = 0.28
    force_contact_ref: float = 35.0
    torque_contact_ref: float = 3.0
    prealign_press_weight: float = 0.02
    prealign_down_xy_gate: float = 0.006
    prealign_down_xy_tau: float = 0.001
    aligned_down_action_weight: float = 0.06
    aligned_up_action_weight: float = 0.04
    depth_delta_ref: float = 0.00025
    depth_delta_reward_weight: float = 0.20
    terminal_depth_window: float = 0.0012
    terminal_depth_weight: float = 0.25
    insert_xy_hold_ref: float = 0.0010
    insert_xy_hold_weight: float = 0.45
    stuck_depth_delta_threshold: float = 0.00002
    stuck_penalty_weight: float = 0.06
    action_penalty_weight: float = 0.010
    time_penalty: float = 0.010
    max_joint_delta: float = 0.04

    # --- Perception perturbation (difficulty / sim-to-real knob) ---
    # Master switch; when False the task behaves exactly as before. Only the
    # observation is corrupted, success and reward always use the true state.
    perturb_enable: bool = False
    # Single scalar that scales every perturbation channel below (e.g. 0.5..2.0).
    perturb_intensity: float = 1.0
    # Scheme 1: per-episode constant vision mis-calibration added to the
    # perceived peg position (m). Forces the policy to find the true hole
    # using touch rather than trusting vision blindly.
    vision_bias_xy_std: float = 0.0015
    vision_bias_z_std: float = 0.0008
    # Per-step gaussian noise on the perceived peg position (m).
    vision_noise_std: float = 0.0005
    # Scheme 2: contact thresholds that occlude (freeze) the vision estimate,
    # so during insertion the policy must rely on force/torque.
    # Force threshold is intentionally high (≥25N) so occlusion only triggers
    # during real peg-in-hole jamming, NOT during light surface approach contact.
    vision_occlusion_force: float = 25.0
    vision_occlusion_depth: float = 0.0005
    # Scheme 4: force/torque sensor noise (N, N*m) and a slow bias random-walk
    # so the touch channel is not a second oracle.
    force_noise_std: float = 2.0
    torque_noise_std: float = 0.2
    force_bias_drift_std: float = 0.3
    force_bias_drift_max: float = 6.0
    torque_bias_drift_std: float = 0.03
    torque_bias_drift_max: float = 0.6

    @property
    def left_offset(self) -> np.ndarray:
        return np.array([-0.5 * self.peg_spacing, 0.0, 0.0], dtype=np.float64)

    @property
    def right_offset(self) -> np.ndarray:
        return np.array([0.5 * self.peg_spacing, 0.0, 0.0], dtype=np.float64)
