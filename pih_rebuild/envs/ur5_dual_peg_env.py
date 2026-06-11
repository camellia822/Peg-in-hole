from __future__ import annotations

from typing import Optional

import gym
import mujoco
import numpy as np
from gym import spaces

from pih_rebuild.config import DualPegTaskConfig
from pih_rebuild.robotics.ur5_kdl import URx_kdl


class UR5DualPegEnv(gym.Env):
    """Clean UR5 IK dual-peg insertion task.

    Policy action is task-space delta xyz. The environment maps it through UR5 IK
    to joint position controls, then observes the dual peg-tip to hole-target
    geometry used by success and reward.
    """

    metadata = {"render_modes": []}

    JOINT_NAMES = (
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    )
    ACTUATOR_NAMES = (
        "shoulder_pan",
        "shoulder_lift",
        "forearm",
        "wrist_1",
        "wrist_2",
        "wrist_3",
    )
    PEG_SITE_NAMES = ("obj_bottom_left", "obj_bottom_right")
    FORCE_SENSOR_NAME = "ee_force_sensor"
    TORQUE_SENSOR_NAME = "ee_torque_sensor"
    EPISODE_REWARD_KEYS = (
        "r_xy_progress",
        "r_xy_distance",
        "r_depth",
        "r_success",
        "r_time",
        "r_stuck",
        "r_action",
        "r_force",
    )
    DONE_REASON_CODES = {
        "success": 0,
        "out_of_workspace": 1,
        "xy_not_aligned": 2,
        "depth_gap": 3,
        "stuck_near_xy": 4,
        "timeout_near_success": 5,
    }

    def __init__(self, config: Optional[DualPegTaskConfig] = None, seed: Optional[int] = None):
        super().__init__()
        self.config = config or DualPegTaskConfig()
        self.rng = np.random.default_rng(seed)

        self.model = mujoco.MjModel.from_xml_path(str(self.config.model_xml))
        self.data = mujoco.MjData(self.model)
        self.ik = URx_kdl(str(self.config.urdf))

        self.joint_ids = np.array([self._joint_id(name) for name in self.JOINT_NAMES], dtype=np.int32)
        self.qpos_addrs = np.array([self.model.jnt_qposadr[jid] for jid in self.joint_ids], dtype=np.int32)
        self.qvel_addrs = np.array([self.model.jnt_dofadr[jid] for jid in self.joint_ids], dtype=np.int32)
        self.actuator_ids = np.array([self._actuator_id(name) for name in self.ACTUATOR_NAMES], dtype=np.int32)
        self.gripper_actuator_id = self._maybe_actuator_id("grippercontrol")
        self.eef_body_id = self._body_id("eef")
        self.box_mocap_id = self._mocap_id("box")
        self.peg_site_ids = np.array([self._site_id(name) for name in self.PEG_SITE_NAMES], dtype=np.int32)
        self.force_sensor_id = self._sensor_id(self.FORCE_SENSOR_NAME)
        self.torque_sensor_id = self._sensor_id(self.TORQUE_SENSOR_NAME)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        obs_dim = 18 + (9 if self.config.use_force_torque_obs else 0)
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32)

        self.fixed_orientation_xyzw = np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.neutral_joint_values = np.deg2rad(np.array([90.0, -90.0, 90.0, -90.0, -90.0, 0.0], dtype=np.float64))
        self.ctrl_low = self.model.actuator_ctrlrange[self.actuator_ids, 0].copy()
        self.ctrl_high = self.model.actuator_ctrlrange[self.actuator_ids, 1].copy()
        self.goal_surface_center = np.array(self.config.hole_surface_center, dtype=np.float64)
        self.ee_target = np.zeros(3, dtype=np.float64)
        self.prev_worst_dist = 0.0
        self.prev_worst_xy_err = 0.0
        self.prev_min_entry_depth = 0.0
        self.force_torque_bias = np.zeros(6, dtype=np.float64)
        self.vision_pos_bias = np.zeros(3, dtype=np.float64)
        self.force_bias_drift = np.zeros(6, dtype=np.float64)
        self._last_perceived_left: Optional[np.ndarray] = None
        self._last_perceived_right: Optional[np.ndarray] = None
        self._last_vision_occluded = False
        self.step_count = 0
        self.episode_diag: dict = {}
        self._prev_phase_idx = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)

        self.goal_surface_center = self._sample_goal_surface_center()
        box_body_pos = self.goal_surface_center.copy()
        box_body_pos[2] -= 0.02
        self.data.mocap_pos[self.box_mocap_id] = box_body_pos
        self.data.mocap_quat[self.box_mocap_id] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        self._set_joint_state(self.neutral_joint_values)
        self.ee_target = self._sample_initial_ee_target()
        q_init = self._ik(self.neutral_joint_values, self.ee_target, limit_delta=False)
        self._set_joint_state(q_init)
        self._set_joint_ctrl(q_init)
        self._set_gripper_ctrl(0.65)
        for _ in range(20):
            mujoco.mj_step(self.model, self.data)
        self.force_torque_bias = self.raw_force_torque()
        self._reset_perturbation_state()

        self.step_count = 0
        reset_metrics = self._xy_depth_metrics()
        self.prev_worst_dist = self._point_metrics()[2]
        self.prev_worst_xy_err = reset_metrics["worst_xy_err"]
        self.prev_min_entry_depth = reset_metrics["min_entry_depth"]
        self._prev_phase_idx = 0
        self._reset_episode_diagnostics()
        return self._get_obs().astype(np.float32)

    def _reset_perturbation_state(self) -> None:
        """Sample per-episode perception perturbations (observation-only)."""
        self.vision_pos_bias = np.zeros(3, dtype=np.float64)
        self.force_bias_drift = np.zeros(6, dtype=np.float64)
        self._last_perceived_left = None
        self._last_perceived_right = None
        self._last_vision_occluded = False
        cfg = self.config
        if not cfg.perturb_enable or cfg.perturb_intensity <= 0.0:
            return
        bias_std = np.array(
            [cfg.vision_bias_xy_std, cfg.vision_bias_xy_std, cfg.vision_bias_z_std],
            dtype=np.float64,
        ) * cfg.perturb_intensity
        self.vision_pos_bias = self.rng.normal(0.0, bias_std)

    def _vision_occluded(self) -> bool:
        """Scheme 2: vision is occluded once the pegs make contact / enter."""
        cfg = self.config
        if cfg.vision_occlusion_force <= 0.0 and cfg.vision_occlusion_depth <= 0.0:
            return False
        force_norm = float(np.linalg.norm(self.current_force_torque()[:3]))
        force_occ = cfg.vision_occlusion_force > 0.0 and force_norm >= cfg.vision_occlusion_force
        depth_occ = (
            cfg.vision_occlusion_depth > 0.0
            and self._min_entry_depth() >= cfg.vision_occlusion_depth
        )
        return bool(force_occ or depth_occ)

    def _perceived_peg_positions(self) -> tuple[np.ndarray, np.ndarray]:
        """Perceived peg tip positions seen by the policy (vision channel).

        Adds a per-episode calibration bias and per-step noise, and freezes the
        estimate while vision is occluded. The true positions returned by
        ``_peg_positions`` are untouched, so success and reward stay exact.
        """
        left_pos, right_pos = self._peg_positions()
        cfg = self.config
        if not cfg.perturb_enable or cfg.perturb_intensity <= 0.0:
            return left_pos, right_pos

        occluded = self._vision_occluded()
        self._last_vision_occluded = occluded
        if occluded and self._last_perceived_left is not None:
            return self._last_perceived_left.copy(), self._last_perceived_right.copy()

        noise_std = cfg.vision_noise_std * cfg.perturb_intensity
        noise_left = self.rng.normal(0.0, noise_std, size=3) if noise_std > 0.0 else np.zeros(3)
        noise_right = self.rng.normal(0.0, noise_std, size=3) if noise_std > 0.0 else np.zeros(3)
        perceived_left = left_pos + self.vision_pos_bias + noise_left
        perceived_right = right_pos + self.vision_pos_bias + noise_right
        self._last_perceived_left = perceived_left.copy()
        self._last_perceived_right = perceived_right.copy()
        return perceived_left, perceived_right

    def _perceived_force_torque(self) -> np.ndarray:
        """Scheme 4: noisy / drifting force-torque seen by the policy."""
        force_torque = self.current_force_torque()
        cfg = self.config
        if not cfg.perturb_enable or cfg.perturb_intensity <= 0.0:
            return force_torque
        intensity = cfg.perturb_intensity
        drift_std = np.array(
            [cfg.force_bias_drift_std] * 3 + [cfg.torque_bias_drift_std] * 3,
            dtype=np.float64,
        ) * intensity
        self.force_bias_drift = self.force_bias_drift + self.rng.normal(0.0, drift_std)
        drift_max = np.array(
            [cfg.force_bias_drift_max] * 3 + [cfg.torque_bias_drift_max] * 3,
            dtype=np.float64,
        )
        self.force_bias_drift = np.clip(self.force_bias_drift, -drift_max, drift_max)
        noise_std = np.array(
            [cfg.force_noise_std] * 3 + [cfg.torque_noise_std] * 3,
            dtype=np.float64,
        ) * intensity
        noise = self.rng.normal(0.0, noise_std) if np.any(noise_std > 0.0) else np.zeros(6)
        return force_torque + self.force_bias_drift + noise

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        action = np.clip(action, -1.0, 1.0)
        xy_action_scale, z_action_scale, action_scale_info = self._action_scales()
        delta = np.array(
            [
                xy_action_scale * action[0],
                xy_action_scale * action[1],
                z_action_scale * action[2],
            ],
            dtype=np.float64,
        )
        self.ee_target = self._clip_ee_target(self.ee_target + delta)

        q_target = self._ik(self.current_joints(), self.ee_target)
        self._set_joint_ctrl(q_target)
        self._set_gripper_ctrl(0.65)
        for _ in range(self.config.control_substeps):
            mujoco.mj_step(self.model, self.data)

        left_dist, right_dist, worst_dist = self._point_metrics()
        success = self._is_success()
        reward, reward_info = self._compute_reward(action, success)

        self.step_count += 1
        self._update_episode_diagnostics(action, reward_info)
        out_of_workspace = self._out_of_workspace()
        terminated = bool(success)
        truncated = self.step_count >= self.config.max_steps or out_of_workspace
        done = terminated or truncated
        done_reason = self._done_reason(success, out_of_workspace, reward_info) if done else ""
        self.prev_worst_dist = worst_dist
        self.prev_min_entry_depth = reward_info["min_entry_depth"]
        force_torque = self.current_force_torque()
        info = {
            "is_success": float(success),
            "left_dist": left_dist,
            "right_dist": right_dist,
            "worst_dist": worst_dist,
            "ee_target": self.ee_target.copy(),
            "ee_force": force_torque[:3],
            "ee_torque": force_torque[3:],
        }
        info.update(reward_info)
        info.update(action_scale_info)
        info["vision_occluded"] = float(self._last_vision_occluded)
        info["vision_bias_norm"] = float(np.linalg.norm(self.vision_pos_bias))
        info["force_bias_norm"] = float(np.linalg.norm(self.force_bias_drift[:3]))
        info.update(self._spar_labels(reward_info))
        if done:
            info.update(self._episode_summary(done_reason, reward_info))
        self.prev_worst_xy_err = reward_info["worst_xy_err"]
        return self._get_obs().astype(np.float32), float(reward), done, info

    def current_joints(self) -> np.ndarray:
        return self.data.qpos[self.qpos_addrs].copy()

    def raw_force_torque(self) -> np.ndarray:
        return np.concatenate(
            [
                self._sensor_data(self.force_sensor_id),
                self._sensor_data(self.torque_sensor_id),
            ]
        )

    def current_force_torque(self) -> np.ndarray:
        return self.raw_force_torque() - self.force_torque_bias

    def _get_obs(self) -> np.ndarray:
        left_pos, right_pos = self._perceived_peg_positions()
        left_goal, right_goal = self._insert_targets()
        left_err = left_goal - left_pos
        right_err = right_goal - right_pos
        metrics = self._xy_depth_metrics(left_err, right_err, left_pos, right_pos)
        obs_parts = [
            left_err / self.config.obs_error_ref,
            right_err / self.config.obs_error_ref,
            np.array(
                [
                    metrics["left_xy_err"] / self.config.obs_xy_ref,
                    metrics["right_xy_err"] / self.config.obs_xy_ref,
                    metrics["worst_xy_err"] / self.config.obs_xy_ref,
                    metrics["sync_xy_err"] / self.config.obs_xy_ref,
                    metrics["left_depth_shortfall"] / self.config.obs_depth_ref,
                    metrics["right_depth_shortfall"] / self.config.obs_depth_ref,
                    metrics["worst_depth_shortfall"] / self.config.obs_depth_ref,
                    metrics["min_entry_depth"] / self.config.obs_depth_ref,
                    metrics["depth_sync_err"] / self.config.obs_depth_ref,
                    metrics["insert_weight"],
                    metrics["inserted_weight"],
                    self.step_count / max(1, self.config.max_steps),
                ],
                dtype=np.float64,
            ),
        ]
        if self.config.use_force_torque_obs:
            force_torque = self._perceived_force_torque()
            force_norm = float(np.linalg.norm(force_torque[:3]))
            torque_norm = float(np.linalg.norm(force_torque[3:]))
            force_contact = self._sigmoid(
                (force_norm - self.config.force_contact_ref) / max(self.config.force_contact_ref, 1e-9)
            )
            torque_contact = self._sigmoid(
                (torque_norm - self.config.torque_contact_ref) / max(self.config.torque_contact_ref, 1e-9)
            )
            obs_parts.append(
                np.concatenate(
                    [
                        force_torque[:3] / self.config.force_obs_ref,
                        force_torque[3:] / self.config.torque_obs_ref,
                        np.array(
                            [
                                force_norm / self.config.force_norm_obs_ref,
                                torque_norm / self.config.torque_norm_obs_ref,
                                max(force_contact, torque_contact),
                            ],
                            dtype=np.float64,
                        ),
                    ]
                )
            )
        obs = np.concatenate(obs_parts)
        return np.clip(obs, -10.0, 10.0)

    def _point_errors(self) -> tuple[np.ndarray, np.ndarray]:
        left_pos, right_pos = self._peg_positions()
        left_goal, right_goal = self._insert_targets()
        return left_goal - left_pos, right_goal - right_pos

    def _point_metrics(self, left_err: Optional[np.ndarray] = None, right_err: Optional[np.ndarray] = None):
        if left_err is None or right_err is None:
            left_err, right_err = self._point_errors()
        left_dist = float(np.linalg.norm(left_err))
        right_dist = float(np.linalg.norm(right_err))
        return left_dist, right_dist, max(left_dist, right_dist)

    def _xy_depth_metrics(
        self,
        left_err: Optional[np.ndarray] = None,
        right_err: Optional[np.ndarray] = None,
        left_pos: Optional[np.ndarray] = None,
        right_pos: Optional[np.ndarray] = None,
    ) -> dict:
        if left_err is None or right_err is None:
            left_err, right_err = self._point_errors()

        if left_pos is None or right_pos is None:
            left_pos, right_pos = self._peg_positions()
        left_xy_err = float(np.linalg.norm(left_err[:2]))
        right_xy_err = float(np.linalg.norm(right_err[:2]))
        worst_xy_err = max(left_xy_err, right_xy_err)
        sync_xy_err = abs(left_xy_err - right_xy_err)

        surface_z = self.goal_surface_center[2]
        target_depth = max(self.config.insertion_depth, 1e-9)
        left_entry_depth = float(surface_z - left_pos[2])
        right_entry_depth = float(surface_z - right_pos[2])
        min_entry_depth = min(left_entry_depth, right_entry_depth)
        left_depth_shortfall = max(target_depth - left_entry_depth, 0.0)
        right_depth_shortfall = max(target_depth - right_entry_depth, 0.0)
        worst_depth_shortfall = max(left_depth_shortfall, right_depth_shortfall)
        depth_sync_err = abs(left_entry_depth - right_entry_depth)
        insert_weight = self._sigmoid((self.config.align_xy_gate - worst_xy_err) / self.config.align_xy_tau)
        inserted_weight = self._sigmoid((min_entry_depth - self.config.inserted_gate) / self.config.inserted_tau)
        xy_excess = max(worst_xy_err - self.config.success_xy_threshold, 0.0)
        depth_closeness = 1.0 - np.clip(
            worst_depth_shortfall / max(self.config.depth_ref, 1e-9),
            0.0,
            1.0,
        )

        return {
            "left_xy_err": left_xy_err,
            "right_xy_err": right_xy_err,
            "worst_xy_err": worst_xy_err,
            "sync_xy_err": sync_xy_err,
            "xy_excess": float(xy_excess),
            "left_entry_depth": left_entry_depth,
            "right_entry_depth": right_entry_depth,
            "min_entry_depth": min_entry_depth,
            "left_depth_shortfall": float(left_depth_shortfall),
            "right_depth_shortfall": float(right_depth_shortfall),
            "worst_depth_shortfall": float(worst_depth_shortfall),
            "depth_closeness": float(depth_closeness),
            "depth_sync_err": float(depth_sync_err),
            "insert_weight": float(insert_weight),
            "inserted_weight": float(inserted_weight),
        }

    def _action_scales(self) -> tuple[float, float, dict]:
        metrics = self._xy_depth_metrics()
        worst_xy_err = metrics["worst_xy_err"]
        worst_depth_shortfall = metrics["worst_depth_shortfall"]
        insert_weight = metrics["insert_weight"]
        depth_near_weight = self._sigmoid(
            (self.config.depth_action_near_gate - worst_depth_shortfall) / self.config.depth_action_tau
        )

        far_den = max(self.config.xy_action_far_gate - self.config.align_xy_gate, 1e-9)
        far_blend = float(np.clip((worst_xy_err - self.config.align_xy_gate) / far_den, 0.0, 1.0))
        preinsert_xy_scale = (
            self.config.xy_action_scale_approach
            + far_blend * (self.config.xy_action_scale_far - self.config.xy_action_scale_approach)
        )
        xy_action_scale = (1.0 - insert_weight) * preinsert_xy_scale + insert_weight * self.config.xy_action_scale_insert
        ready_z_scale = (
            self.config.z_action_scale_insert
            + (1.0 - depth_near_weight) * (self.config.z_action_scale_far - self.config.z_action_scale_insert)
        )
        z_action_scale = (1.0 - insert_weight) * self.config.z_action_scale_prealign + insert_weight * ready_z_scale
        return (
            float(xy_action_scale),
            float(z_action_scale),
            {
                "action_xy_scale": float(xy_action_scale),
                "action_z_scale": float(z_action_scale),
                "action_insert_weight": float(insert_weight),
                "action_depth_near_weight": float(depth_near_weight),
            },
        )

    def _reset_episode_diagnostics(self) -> None:
        self.episode_diag = {
            "steps": 0,
            "sum_e_xy": 0.0,
            "sum_depth_gap": 0.0,
            "sum_depth_delta": 0.0,
            "near_xy_steps": 0,
            "stuck_steps": 0,
            "sum_action_norm": 0.0,
            "sum_action_norm_near_xy": 0.0,
            "reward_sums": {key: 0.0 for key in self.EPISODE_REWARD_KEYS},
        }

    def _update_episode_diagnostics(self, action: np.ndarray, reward_info: dict) -> None:
        diag = self.episode_diag
        diag["steps"] += 1
        e_xy = float(reward_info["worst_xy_err"])
        depth_gap = float(reward_info["worst_depth_shortfall"])
        depth_delta = float(reward_info["depth_delta"])
        action_norm = float(np.linalg.norm(action))
        near_xy = e_xy <= self.config.align_xy_gate
        stuck = (
            near_xy
            and depth_gap > self.config.success_depth_tolerance
            and abs(depth_delta) <= self.config.stuck_depth_delta_threshold
        )

        diag["sum_e_xy"] += e_xy
        diag["sum_depth_gap"] += depth_gap
        diag["sum_depth_delta"] += depth_delta
        diag["sum_action_norm"] += action_norm
        if near_xy:
            diag["near_xy_steps"] += 1
            diag["sum_action_norm_near_xy"] += action_norm
        if stuck:
            diag["stuck_steps"] += 1
        for key in self.EPISODE_REWARD_KEYS:
            diag["reward_sums"][key] += float(reward_info[f"step_{key}"])

    def _done_reason(self, success: bool, out_of_workspace: bool, reward_info: dict) -> str:
        if success:
            return "success"
        if out_of_workspace:
            return "out_of_workspace"

        steps = max(int(self.episode_diag["steps"]), 1)
        near_xy_rate = float(self.episode_diag["near_xy_steps"]) / steps
        stuck_rate = float(self.episode_diag["stuck_steps"]) / steps
        if near_xy_rate > 0.5 and stuck_rate > 0.5:
            return "stuck_near_xy"
        if float(reward_info["worst_xy_err"]) > self.config.align_xy_gate:
            return "xy_not_aligned"
        if float(reward_info["worst_depth_shortfall"]) > self.config.success_depth_tolerance:
            return "depth_gap"
        return "timeout_near_success"

    def _episode_summary(self, done_reason: str, reward_info: dict) -> dict:
        diag = self.episode_diag
        steps = max(int(diag["steps"]), 1)
        near_xy_steps = int(diag["near_xy_steps"])
        failure_reason = "none" if done_reason == "success" else done_reason
        summary = {
            "episode_steps": float(steps),
            "e_xy_mean": float(diag["sum_e_xy"] / steps),
            "final_e_xy": float(reward_info["worst_xy_err"]),
            "depth_gap_mean": float(diag["sum_depth_gap"] / steps),
            "final_depth_gap": float(reward_info["worst_depth_shortfall"]),
            "delta_depth_mean": float(diag["sum_depth_delta"] / steps),
            "near_xy_rate": float(near_xy_steps / steps),
            "near_xy_steps": float(near_xy_steps),
            "stuck_rate": float(diag["stuck_steps"] / steps),
            "done_reason": done_reason,
            "done_reason_code": float(self.DONE_REASON_CODES.get(done_reason, -1)),
            "failure_reason": failure_reason,
            "failure_reason_code": float(self.DONE_REASON_CODES.get(failure_reason, -1)) if failure_reason != "none" else 0.0,
            "action_norm": float(diag["sum_action_norm"] / steps),
            "action_norm_near_xy": float(diag["sum_action_norm_near_xy"] / max(near_xy_steps, 1)),
        }
        summary.update({key: float(diag["reward_sums"][key]) for key in self.EPISODE_REWARD_KEYS})
        return summary

    def _spar_labels(self, reward_info: dict) -> dict:
        """Environment-side auxiliary phase signals for the SPAR M1/M2 modules.

        Only written into the ``info`` dict; observation, reward, action and
        physics are left untouched, so a plain-SAC run on this env is identical
        to the version without these labels. The four phases are
        ``search / align / insert / recovery`` where ``recovery`` is redefined
        for this task as "aligned but insertion has stalled" (the dominant
        depth_gap failure), so every phase carries real samples.
        """
        cfg = self.config
        e_xy = float(reward_info["worst_xy_err"])
        dsf = float(reward_info["worst_depth_shortfall"])
        ddelta = float(reward_info["depth_delta"])
        sync_err = float(reward_info["depth_sync_err"])
        xy_progress = float(reward_info["xy_progress"])

        coarse = self._sigmoid((e_xy - cfg.prealign_down_xy_gate) / 0.002)
        aligned = self._sigmoid((cfg.align_xy_gate - e_xy) / max(cfg.align_xy_tau, 1e-9))
        not_inserted = self._sigmoid((dsf - cfg.success_depth_tolerance) / 0.0008)
        stalled = self._sigmoid(
            (cfg.stuck_depth_delta_threshold - ddelta) / max(cfg.stuck_depth_delta_threshold, 1e-9)
        )
        stalled = stalled * not_inserted

        search = coarse
        align = (1.0 - coarse) * (1.0 - aligned)
        insert = aligned * (1.0 - stalled)
        recovery = aligned * stalled
        raw = np.array([search, align, insert, recovery], dtype=np.float64)
        total = float(raw.sum())
        if total < 1e-8:
            probs = np.full(4, 0.25, dtype=np.float64)
        else:
            probs = raw / total

        phase_idx = int(np.argmax(probs))
        switch = 1.0 if phase_idx != self._prev_phase_idx else 0.0
        self._prev_phase_idx = phase_idx

        return {
            "spar_phase_target": probs.astype(np.float32),
            "spar_switch_target": float(switch),
            "spar_stall_score": float(aligned * stalled),
            "spar_sync_score": float(np.tanh(sync_err / max(cfg.depth_sync_ref, 1e-9))),
            "spar_align_progress": float(np.clip(xy_progress, 0.0, 1.0)),
            "spar_insert_progress": float(np.clip(ddelta / max(cfg.depth_delta_ref, 1e-9), 0.0, 1.0)),
            "spar_valid": 1.0,
            "spar_phase_idx": float(phase_idx),
        }

    def _peg_positions(self) -> tuple[np.ndarray, np.ndarray]:
        left = self.data.site_xpos[self.peg_site_ids[0]].copy()
        right = self.data.site_xpos[self.peg_site_ids[1]].copy()
        return left, right

    def _insert_targets(self) -> tuple[np.ndarray, np.ndarray]:
        center = self.goal_surface_center.copy()
        center[2] -= self.config.insertion_depth
        return center + self.config.left_offset, center + self.config.right_offset

    def _is_success(self) -> bool:
        metrics = self._xy_depth_metrics()
        depth_ok = metrics["min_entry_depth"] >= (
            self.config.insertion_depth - self.config.success_depth_tolerance
        )
        xy_ok = metrics["worst_xy_err"] <= self.config.success_xy_threshold
        return bool(xy_ok and depth_ok)

    def _min_entry_depth(self) -> float:
        left_pos, right_pos = self._peg_positions()
        surface_z = self.goal_surface_center[2]
        return min(float(surface_z - left_pos[2]), float(surface_z - right_pos[2]))

    def _compute_reward(self, action: np.ndarray, success: bool) -> tuple[float, dict]:
        left_pos, right_pos = self._peg_positions()
        left_goal, right_goal = self._insert_targets()
        left_err = left_goal - left_pos
        right_err = right_goal - right_pos
        metrics = self._xy_depth_metrics(left_err, right_err)

        left_xy_err = metrics["left_xy_err"]
        right_xy_err = metrics["right_xy_err"]
        worst_xy_err = metrics["worst_xy_err"]
        sync_xy_err = metrics["sync_xy_err"]
        insert_weight = metrics["insert_weight"]
        inserted_weight = metrics["inserted_weight"]
        left_entry_depth = metrics["left_entry_depth"]
        right_entry_depth = metrics["right_entry_depth"]
        min_entry_depth = metrics["min_entry_depth"]
        depth_delta = min_entry_depth - self.prev_min_entry_depth
        target_depth = max(self.config.insertion_depth, 1e-9)
        worst_depth_shortfall = metrics["worst_depth_shortfall"]
        depth_sync_err = metrics["depth_sync_err"]

        xy_delta = self.prev_worst_xy_err - worst_xy_err
        xy_progress = float(np.clip(xy_delta / self.config.xy_progress_ref, -1.0, 1.0))
        prev_insert_weight = self._sigmoid(
            (self.config.align_xy_gate - self.prev_worst_xy_err) / self.config.align_xy_tau
        )
        xy_ready_delta = insert_weight - prev_insert_weight
        xy_ready_progress = float(np.clip(xy_ready_delta / 0.25, -1.0, 1.0))
        entry_progress = float(np.clip(min_entry_depth / target_depth, 0.0, 1.0))
        depth_closeness = metrics["depth_closeness"]
        depth_score = 2.0 * depth_closeness - 1.0
        terminal_depth_progress = 1.0 - np.clip(
            worst_depth_shortfall / self.config.terminal_depth_window,
            0.0,
            1.0,
        )

        r_xy_progress = (
            self.config.xy_progress_weight * xy_progress
            + self.config.xy_ready_bonus_weight * xy_ready_progress
        )
        xy_stage_weight = 1.05 - 0.25 * insert_weight
        r_xy_distance = -xy_stage_weight * (
            0.80 * np.tanh(worst_xy_err / self.config.align_xy_ref)
            + 0.20 * np.tanh(sync_xy_err / self.config.align_sync_ref)
        )

        r_depth_distance = insert_weight * self.config.depth_distance_weight * depth_score
        r_depth_level = insert_weight * self.config.depth_level_weight * entry_progress
        r_depth_sync = -0.12 * insert_weight * np.tanh(depth_sync_err / self.config.depth_sync_ref)
        r_insert = r_depth_distance + r_depth_level + r_depth_sync
        r_terminal_depth = insert_weight * self.config.terminal_depth_weight * terminal_depth_progress
        xy_excess = metrics["xy_excess"]
        xy_hold_phase = max(insert_weight, inserted_weight)
        r_insert_xy_hold = -self.config.insert_xy_hold_weight * xy_hold_phase * np.tanh(
            xy_excess / self.config.insert_xy_hold_ref
        )

        prealign_press = max(-float(action[2]), 0.0)
        xy_far_down_weight = self._sigmoid(
            (worst_xy_err - self.config.prealign_down_xy_gate) / self.config.prealign_down_xy_tau
        )
        r_prealign_press = -self.config.prealign_press_weight * xy_far_down_weight * prealign_press
        aligned_down_action = max(-float(action[2]), 0.0)
        aligned_up_action = max(float(action[2]), 0.0)
        depth_need = float(np.clip(worst_depth_shortfall / target_depth, 0.0, 1.0))
        r_aligned_z_action = insert_weight * (
            self.config.aligned_down_action_weight * aligned_down_action * depth_need
            - self.config.aligned_up_action_weight * aligned_up_action
        )
        r_depth_delta = insert_weight * self.config.depth_delta_reward_weight * np.clip(
            depth_delta / self.config.depth_delta_ref,
            -1.0,
            1.0,
        )
        stuck = (
            insert_weight > 0.65
            and worst_depth_shortfall > self.config.success_depth_tolerance
            and abs(depth_delta) <= self.config.stuck_depth_delta_threshold
        )
        r_stuck = -self.config.stuck_penalty_weight if stuck else 0.0

        force_torque = self.current_force_torque()
        force_norm = float(np.linalg.norm(force_torque[:3]))
        torque_norm = float(np.linalg.norm(force_torque[3:]))
        contact_weight = max(insert_weight, inserted_weight)
        force_weight = self.config.force_weight_preinsert + contact_weight * (
            self.config.force_weight_inserted - self.config.force_weight_preinsert
        )
        force_excess = max(force_norm - self.config.force_safe, 0.0) / self.config.force_penalty_ref
        torque_excess = max(torque_norm - self.config.torque_safe, 0.0) / self.config.torque_penalty_ref
        r_force = -force_weight * min(force_excess * force_excess, 5.0)
        r_force -= 0.5 * force_weight * min(torque_excess * torque_excess, 5.0)

        r_action = -self.config.action_penalty_weight * float(np.dot(action, action))
        r_time = -self.config.time_penalty
        r_success = self.config.success_bonus if success else 0.0
        r_depth = r_insert + r_terminal_depth + r_prealign_press + r_aligned_z_action + r_depth_delta
        reward = (
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
        )

        info = {
            "reward_align": float(r_xy_progress + r_xy_distance),
            "reward_xy_progress": float(r_xy_progress),
            "reward_xy_distance": float(r_xy_distance),
            "reward_insert": float(r_insert),
            "reward_depth_distance": float(r_depth_distance),
            "reward_depth_level": float(r_depth_level),
            "reward_depth_sync": float(r_depth_sync),
            "reward_terminal_depth": float(r_terminal_depth),
            "reward_insert_xy_hold": float(r_insert_xy_hold),
            "reward_force": float(r_force),
            "reward_prealign_press": float(r_prealign_press),
            "reward_aligned_z_action": float(r_aligned_z_action),
            "reward_depth_delta": float(r_depth_delta),
            "reward_action": float(r_action),
            "reward_time": float(r_time),
            "reward_success": float(r_success),
            "step_r_xy_progress": float(r_xy_progress),
            "step_r_xy_distance": float(r_xy_distance + r_insert_xy_hold),
            "step_r_depth": float(r_depth),
            "step_r_success": float(r_success),
            "step_r_time": float(r_time),
            "step_r_stuck": float(r_stuck),
            "step_r_action": float(r_action),
            "step_r_force": float(r_force),
            "insert_weight": float(insert_weight),
            "inserted_weight": float(inserted_weight),
            "left_xy_err": left_xy_err,
            "right_xy_err": right_xy_err,
            "sync_xy_err": float(sync_xy_err),
            "worst_xy_err": worst_xy_err,
            "xy_excess": float(xy_excess),
            "xy_ready_delta": float(xy_ready_delta),
            "xy_ready_progress": float(xy_ready_progress),
            "left_entry_depth": left_entry_depth,
            "right_entry_depth": right_entry_depth,
            "min_entry_depth": min_entry_depth,
            "depth_delta": float(depth_delta),
            "left_depth_shortfall": metrics["left_depth_shortfall"],
            "right_depth_shortfall": metrics["right_depth_shortfall"],
            "worst_depth_shortfall": float(worst_depth_shortfall),
            "depth_closeness": float(depth_closeness),
            "depth_score": float(depth_score),
            "terminal_depth_progress": float(terminal_depth_progress),
            "entry_progress": float(entry_progress),
            "xy_progress": float(xy_progress),
            "xy_far_down_weight": float(xy_far_down_weight),
            "depth_sync_err": float(depth_sync_err),
            "force_norm": force_norm,
            "torque_norm": torque_norm,
            "force_weight": float(force_weight),
            "force_excess": float(force_excess),
            "torque_excess": float(torque_excess),
        }
        return float(reward), info

    @staticmethod
    def _sigmoid(value: float) -> float:
        value = float(np.clip(value, -60.0, 60.0))
        return float(1.0 / (1.0 + np.exp(-value)))

    def _sample_goal_surface_center(self) -> np.ndarray:
        center = np.array(self.config.hole_surface_center, dtype=np.float64)
        low, high = self.config.goal_xy_random
        if high > low:
            center[:2] += self.rng.uniform(low, high, size=2)
        return center

    def _sample_initial_ee_target(self) -> np.ndarray:
        low, high = self.config.init_xy_random
        target = self.goal_surface_center.copy()
        target[:2] += self.rng.uniform(low, high, size=2)
        target[2] += self.config.init_height_above_surface
        return target

    def _clip_ee_target(self, target: np.ndarray) -> np.ndarray:
        low = self.goal_surface_center + np.array([-0.08, -0.08, -0.02], dtype=np.float64)
        high = self.goal_surface_center + np.array([0.08, 0.08, 0.14], dtype=np.float64)
        return np.clip(target, low, high)

    def _ik(self, current_joint: np.ndarray, target_position: np.ndarray, limit_delta: bool = True) -> np.ndarray:
        q = np.asarray(self.ik.inverse(current_joint, target_position, self.fixed_orientation_xyzw), dtype=np.float64)
        if q.shape[0] != 6 or not np.all(np.isfinite(q)):
            return current_joint.copy()
        q = current_joint + (q - current_joint + np.pi) % (2.0 * np.pi) - np.pi
        if limit_delta:
            delta = np.clip(q - current_joint, -self.config.max_joint_delta, self.config.max_joint_delta)
            q = current_joint + delta
        return np.clip(q, self.ctrl_low, self.ctrl_high)

    def _set_joint_state(self, q: np.ndarray) -> None:
        self.data.qpos[self.qpos_addrs] = q
        self.data.qvel[self.qvel_addrs] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _set_joint_ctrl(self, q: np.ndarray) -> None:
        self.data.ctrl[self.actuator_ids] = q

    def _set_gripper_ctrl(self, value: float) -> None:
        if self.gripper_actuator_id is not None:
            self.data.ctrl[self.gripper_actuator_id] = value

    def _out_of_workspace(self) -> bool:
        peg_center = 0.5 * (self._peg_positions()[0] + self._peg_positions()[1])
        return bool(np.linalg.norm(peg_center[:2] - self.goal_surface_center[:2]) > 0.12)

    def _body_id(self, name: str) -> int:
        idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if idx < 0:
            raise KeyError(name)
        return idx

    def _joint_id(self, name: str) -> int:
        idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if idx < 0:
            raise KeyError(name)
        return idx

    def _actuator_id(self, name: str) -> int:
        idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if idx < 0:
            raise KeyError(name)
        return idx

    def _maybe_actuator_id(self, name: str) -> Optional[int]:
        idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        return None if idx < 0 else idx

    def _site_id(self, name: str) -> int:
        idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        if idx < 0:
            raise KeyError(name)
        return idx

    def _sensor_id(self, name: str) -> int:
        idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if idx < 0:
            raise KeyError(name)
        return idx

    def _sensor_data(self, sensor_id: int) -> np.ndarray:
        adr = self.model.sensor_adr[sensor_id]
        dim = self.model.sensor_dim[sensor_id]
        return self.data.sensordata[adr : adr + dim].copy()

    def _mocap_id(self, body_name: str) -> int:
        body_id = self._body_id(body_name)
        mocap_id = int(self.model.body_mocapid[body_id])
        if mocap_id < 0:
            raise KeyError(f"Body {body_name!r} is not a mocap body")
        return mocap_id
