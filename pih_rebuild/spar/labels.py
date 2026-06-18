"""SPAR-SAC M1: environment-side soft phase & event signal generator.

This module turns the dense ``reward_info`` already produced by
``UR5DualPegEnv._compute_reward`` into the auxiliary signals required by the
three SPAR-SAC modules described in the patent:

* a four-dimensional *soft* phase probability vector
  ``[search, align, insert, recovery]`` (M1 supervision target),
* a scalar phase-boundary *switch* confidence (M1 supervision target),
* a set of task-aware event signals (contact criticality, near-success,
  recovery, dual-axis synchronisation, alignment / insertion progress) that
  M3's asymmetric prioritised replay consumes.

It writes **only** to the ``info`` dict returned by ``env.step``; it never
touches the observation, reward, action or physics, so a baseline SAC run that
ignores the ``spar_*`` keys stays bit-for-bit comparable.

The phase definition deliberately re-grounds ``recovery`` as *"aligned but the
insertion has stalled"* (the dominant ``depth_gap`` failure mode of this task)
so that every one of the four phases is populated by real transitions instead
of being a near-empty class.
"""

from __future__ import annotations

import numpy as np

PHASE_NAMES = ("search", "align", "insert", "recovery")


def _sigmoid(x: float) -> float:
    x = float(np.clip(x, -60.0, 60.0))
    return float(1.0 / (1.0 + np.exp(-x)))


class SparLabeler:
    """Stateful per-episode generator of SPAR phase/event signals.

    The labeler keeps a small amount of cross-step state (previous force
    magnitude, whether the previous step was stalled) so it can emit
    *force-jump* and *recovery* event signals. Call :meth:`reset` at every
    ``env.reset`` and :meth:`compute` once per ``env.step``.
    """

    def __init__(self, config):
        self.config = config
        # Soft-gate length scales (metres) derived from the task tolerances so
        # they automatically track ``--insertion_depth_mm`` style overrides.
        gate = float(config.align_xy_gate)
        self._search_lo = 6.0 * gate
        self._search_tau = 2.0 * gate
        self._align_gate = gate
        self._align_tau = float(config.align_xy_tau)
        self._depth_tol = float(config.success_depth_tolerance)
        self._depth_tol_tau = max(0.04 * float(config.insertion_depth), 1e-4)
        self._stuck_thr = float(config.stuck_depth_delta_threshold)
        self._depth_delta_ref = float(config.depth_delta_ref)
        self._force_contact_ref = float(config.force_contact_ref)
        self._force_jump_thr = 0.5 * float(config.force_contact_ref)
        self._sync_xy_ref = float(config.align_sync_ref)
        self._depth_sync_ref = float(config.depth_sync_ref)
        self._success_xy = float(config.success_xy_threshold)
        self._insertion_depth = max(float(config.insertion_depth), 1e-9)
        self.reset()

    def reset(self) -> None:
        self._prev_force_norm: float | None = None
        self._was_stalled = False

    def compute(self, reward_info: dict, success: bool) -> dict:
        e_xy = float(reward_info["worst_xy_err"])
        dsf = float(reward_info["worst_depth_shortfall"])
        sync_xy = float(reward_info["sync_xy_err"])
        depth_sync = float(reward_info["depth_sync_err"])
        ddelta = float(reward_info["depth_delta"])
        min_depth = float(reward_info["min_entry_depth"])
        force_norm = float(reward_info["force_norm"])
        insert_weight = float(reward_info["insert_weight"])
        xy_progress = float(reward_info.get("xy_progress", 0.0))

        # ---- soft phase membership ---------------------------------------
        coarse = _sigmoid((e_xy - self._search_lo) / self._search_tau)
        aligned = _sigmoid((self._align_gate - e_xy) / self._align_tau)
        not_inserted = _sigmoid((dsf - self._depth_tol) / self._depth_tol_tau)
        stalled = (
            aligned
            * not_inserted
            * _sigmoid((self._stuck_thr - ddelta) / max(self._stuck_thr, 1e-9))
        )

        search = coarse
        align = (1.0 - coarse) * (1.0 - aligned)
        insert = aligned * (1.0 - stalled)
        recovery = aligned * stalled

        phase = np.array([search, align, insert, recovery], dtype=np.float64)
        total = float(phase.sum())
        if total > 1e-8:
            phase = phase / total
        else:
            phase = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        phase_idx = int(np.argmax(phase))
        # Switch confidence is high near a phase boundary, i.e. when no single
        # phase dominates the membership distribution.
        switch_target = float(np.clip(1.0 - float(phase.max()), 0.0, 1.0))

        # ---- M3 task-aware event signals ---------------------------------
        contact = _sigmoid(
            (force_norm - self._force_contact_ref) / max(self._force_contact_ref, 1e-9)
        )
        if self._prev_force_norm is None:
            force_jump = 0.0
        else:
            force_jump = _sigmoid(
                (abs(force_norm - self._prev_force_norm) - self._force_jump_thr)
                / max(self._force_jump_thr, 1e-9)
            )
        sync_gap = 0.5 * float(np.clip(sync_xy / max(self._sync_xy_ref, 1e-9), 0.0, 1.0))
        sync_gap += 0.5 * float(
            np.clip(depth_sync / max(self._depth_sync_ref, 1e-9), 0.0, 1.0)
        )
        align_progress = float(np.clip(xy_progress, 0.0, 1.0))
        insert_progress = float(
            np.clip(ddelta / max(self._depth_delta_ref, 1e-9), 0.0, 1.0)
        )
        near_success = float(
            (e_xy < self._success_xy)
            and (min_depth >= 0.8 * self._insertion_depth)
            and (not success)
        )
        recovery_event = float(self._was_stalled and stalled < 0.5)

        self._prev_force_norm = force_norm
        self._was_stalled = stalled >= 0.5

        return {
            "spar_phase_target": phase.astype(np.float32),
            "spar_switch_target": switch_target,
            "spar_phase_idx": phase_idx,
            "spar_valid": 1.0,
            "spar_contact": float(contact),
            "spar_force_jump": float(force_jump),
            "spar_sync_gap": float(sync_gap),
            "spar_align_progress": float(align_progress),
            "spar_insert_progress": float(insert_progress),
            "spar_stall": float(stalled),
            "spar_near_success": float(near_success),
            "spar_recovery": float(recovery_event),
            "spar_p_search": float(phase[0]),
            "spar_p_align": float(phase[1]),
            "spar_p_insert": float(phase[2]),
            "spar_p_recovery": float(phase[3]),
        }
