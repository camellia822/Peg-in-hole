"""SPAR-SAC M3: asymmetric prioritised experience replay.

``SPARReplayBuffer`` extends the standard SB3 :class:`ReplayBuffer` with the
extra per-step signals emitted by :mod:`pih_rebuild.spar.labels` (soft phase
target, switch target and the task-aware event signals) and implements the
patent's asymmetric prioritised replay:

* a task-aware *base* priority built from contact-criticality, near-success,
  recovery-value and dual-axis synchronisation scores;
* an *event window* boost that exponentially up-weights the transitions leading
  into a key event (contact burst, near-success, recovery, stall);
* a TD-error driven priority that is shared with, but treated *asymmetrically*
  by, the actor and the critic -- the critic uses the full TD magnitude to
  accelerate value correction, while the actor clips it so extreme failure
  samples cannot dominate the policy gradient;
* per-stream importance-sampling weights that de-bias the non-uniform sampling.

Even with M3 disabled the buffer is used (so M1/M2 can read the stored phase
labels); in that case the algorithm simply calls :meth:`sample`, which performs
ordinary uniform sampling and returns unit IS weights -- numerically identical
to vanilla SAC replay.
"""

from __future__ import annotations

from typing import Any, Dict, List, NamedTuple, Optional, Union

import numpy as np
import torch as th
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import VecNormalize


class SPARReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    phase_target: th.Tensor
    switch_target: th.Tensor
    align_progress: th.Tensor
    insert_progress: th.Tensor
    stall: th.Tensor
    sync_gap: th.Tensor
    is_weights: th.Tensor
    batch_inds: np.ndarray
    env_indices: np.ndarray


class SPARReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space,
        action_space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        n_phases: int = 4,
        alpha: float = 0.6,
        beta_is: float = 0.4,
        eps: float = 1e-2,
        td_cap: float = 2.0,
        w_contact: float = 0.5,
        w_near: float = 0.5,
        w_recovery: float = 0.5,
        w_sync: float = 0.3,
        window_radius: int = 5,
        window_decay: float = 2.0,
        event_boost: float = 0.5,
    ):
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination,
        )
        self.n_phases = int(n_phases)
        self.alpha = float(alpha)
        self.beta_is = float(beta_is)
        self.eps = float(eps)
        self.td_cap = float(td_cap)
        self.w_contact = float(w_contact)
        self.w_near = float(w_near)
        self.w_recovery = float(w_recovery)
        self.w_sync = float(w_sync)
        self.window_radius = int(window_radius)
        self.window_decay = float(window_decay)
        self.event_boost = float(event_boost)

        shape = (self.buffer_size, self.n_envs)
        self.phase_target = np.zeros((*shape, self.n_phases), dtype=np.float32)
        self.phase_target[..., 0] = 1.0  # default = pure "search"
        self.switch_target = np.zeros(shape, dtype=np.float32)
        self.align_progress = np.zeros(shape, dtype=np.float32)
        self.insert_progress = np.zeros(shape, dtype=np.float32)
        self.stall = np.zeros(shape, dtype=np.float32)
        self.sync_gap = np.zeros(shape, dtype=np.float32)
        self.contact = np.zeros(shape, dtype=np.float32)
        self.force_jump = np.zeros(shape, dtype=np.float32)
        self.near_success = np.zeros(shape, dtype=np.float32)
        self.recovery = np.zeros(shape, dtype=np.float32)
        # Task-aware base priority and TD-error priority.
        self.base_priority = np.ones(shape, dtype=np.float32)
        self.td_abs = np.ones(shape, dtype=np.float32)
        self._max_td = 1.0

    # ------------------------------------------------------------------
    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        pos = self.pos  # capture before the base class advances the cursor
        super().add(obs, next_obs, action, reward, done, infos)
        for env_idx, info in enumerate(infos):
            phase = info.get("spar_phase_target")
            if phase is None:
                phase = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            self.phase_target[pos, env_idx] = np.asarray(phase, dtype=np.float32)
            self.switch_target[pos, env_idx] = float(info.get("spar_switch_target", 0.0))
            self.align_progress[pos, env_idx] = float(info.get("spar_align_progress", 0.0))
            self.insert_progress[pos, env_idx] = float(info.get("spar_insert_progress", 0.0))
            self.stall[pos, env_idx] = float(info.get("spar_stall", 0.0))
            self.sync_gap[pos, env_idx] = float(info.get("spar_sync_gap", 0.0))
            contact = float(info.get("spar_contact", 0.0))
            force_jump = float(info.get("spar_force_jump", 0.0))
            near = float(info.get("spar_near_success", 0.0))
            recovery = float(info.get("spar_recovery", 0.0))
            stall = self.stall[pos, env_idx]
            self.contact[pos, env_idx] = contact
            self.force_jump[pos, env_idx] = force_jump
            self.near_success[pos, env_idx] = near
            self.recovery[pos, env_idx] = recovery

            # Patent priority scores (eq. 7-10): contact criticality, near
            # success, recovery value, dual-axis synchronisation.
            p_contact = 0.5 * contact + 0.3 * force_jump + 0.2 * self.sync_gap[pos, env_idx]
            base = (
                1.0
                + self.w_contact * p_contact
                + self.w_near * near
                + self.w_recovery * recovery
                + self.w_sync * self.sync_gap[pos, env_idx]
            )
            self.base_priority[pos, env_idx] = base
            self.td_abs[pos, env_idx] = self._max_td  # new sample -> max priority

            is_event = (
                contact > 0.5
                or force_jump > 0.5
                or near > 0.5
                or recovery > 0.5
                or stall > 0.5
            )
            if is_event and self.window_radius > 0:
                self._boost_event_window(pos, env_idx)

    def _boost_event_window(self, pos: int, env_idx: int) -> None:
        """Exponentially up-weight the transitions leading into a key event."""
        filled = self.buffer_size if self.full else pos
        for d in range(1, self.window_radius + 1):
            if d > filled:
                break
            idx = (pos - d) % self.buffer_size
            boost = self.event_boost * float(np.exp(-d / max(self.window_decay, 1e-9)))
            self.base_priority[idx, env_idx] *= 1.0 + boost

    # ------------------------------------------------------------------
    def _make_samples(
        self,
        batch_inds: np.ndarray,
        env_indices: np.ndarray,
        is_weights: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> SPARReplayBufferSamples:
        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(
                self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env
            )
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        dones = (
            self.dones[batch_inds, env_indices]
            * (1 - self.timeouts[batch_inds, env_indices])
        ).reshape(-1, 1)

        return SPARReplayBufferSamples(
            observations=self.to_torch(self._normalize_obs(self.observations[batch_inds, env_indices, :], env)),
            actions=self.to_torch(self.actions[batch_inds, env_indices, :]),
            next_observations=self.to_torch(next_obs),
            dones=self.to_torch(dones),
            rewards=self.to_torch(self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env)),
            phase_target=self.to_torch(self.phase_target[batch_inds, env_indices, :]),
            switch_target=self.to_torch(self.switch_target[batch_inds, env_indices].reshape(-1, 1)),
            align_progress=self.to_torch(self.align_progress[batch_inds, env_indices].reshape(-1, 1)),
            insert_progress=self.to_torch(self.insert_progress[batch_inds, env_indices].reshape(-1, 1)),
            stall=self.to_torch(self.stall[batch_inds, env_indices].reshape(-1, 1)),
            sync_gap=self.to_torch(self.sync_gap[batch_inds, env_indices].reshape(-1, 1)),
            is_weights=self.to_torch(is_weights.reshape(-1, 1).astype(np.float32)),
            batch_inds=batch_inds,
            env_indices=env_indices,
        )

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> SPARReplayBufferSamples:
        # Uniform sampling path (used for M1/M2 and as the M3-disabled default).
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))
        is_weights = np.ones(len(batch_inds), dtype=np.float32)
        return self._make_samples(batch_inds, env_indices, is_weights, env)

    def sample_prioritized(
        self,
        batch_size: int,
        stream: str = "critic",
        env: Optional[VecNormalize] = None,
    ) -> SPARReplayBufferSamples:
        """Asymmetric prioritised sampling for the critic or the actor stream."""
        upper = self.buffer_size if self.full else self.pos
        base = self.base_priority[:upper]
        td = self.td_abs[:upper]
        if stream == "actor":
            td = np.minimum(td, self.td_cap)
        prio = np.power(td + self.eps, self.alpha) * base
        flat = prio.reshape(-1)
        total = float(flat.sum())
        if total <= 0.0 or not np.isfinite(total):
            probs = np.full(flat.shape, 1.0 / flat.size, dtype=np.float64)
        else:
            probs = (flat / total).astype(np.float64)
        # Inverse-CDF sampling avoids np.random.choice's strict "sum to 1"
        # check, which trips on harmless float round-off in the normalised probs.
        cdf = np.cumsum(probs)
        cdf[-1] = 1.0
        idx = np.searchsorted(cdf, np.random.random_sample(batch_size), side="right")
        idx = np.clip(idx, 0, flat.size - 1).astype(np.int64)
        batch_inds = (idx // self.n_envs).astype(np.int64)
        env_indices = (idx % self.n_envs).astype(np.int64)

        n = flat.size
        sample_p = probs[idx]
        is_weights = np.power(np.maximum(n * sample_p, 1e-12), -self.beta_is)
        is_weights = is_weights / is_weights.max()
        return self._make_samples(batch_inds, env_indices, is_weights, env)

    def update_priorities(self, batch_inds: np.ndarray, env_indices: np.ndarray, td_abs: np.ndarray) -> None:
        td_abs = np.abs(np.asarray(td_abs, dtype=np.float32)).reshape(-1) + self.eps
        self.td_abs[batch_inds, env_indices] = td_abs
        if td_abs.size > 0:
            self._max_td = float(max(self._max_td, td_abs.max()))
