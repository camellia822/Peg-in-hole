"""Replay buffer that carries the SPAR auxiliary phase labels from env infos.

The buffer is a thin, SB3-compatible extension of ``ReplayBuffer``: it stores
the same transitions plus per-step phase labels emitted by the environment in
``info`` (``spar_phase_target``, ``spar_switch_target``, ``spar_stall_score``,
``spar_sync_score``, ``spar_align_progress``, ``spar_insert_progress``,
``spar_valid``, ``spar_phase_idx``). Transitions without those keys (e.g. a
plain-SAC env) get ``spar_valid = 0`` so the auxiliary losses can mask them out.
"""

from typing import Any, Dict, List, NamedTuple, Optional

import numpy as np
import torch as th

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize


class SPARReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    phase_targets: th.Tensor
    switch_targets: th.Tensor
    spar_valid: th.Tensor
    phase_labels: th.Tensor
    stall_scores: th.Tensor
    sync_scores: th.Tensor
    align_progresses: th.Tensor
    insert_progresses: th.Tensor


class SPARReplayBuffer(ReplayBuffer):
    """``ReplayBuffer`` that preserves SPAR auxiliary labels from env infos."""

    def __init__(self, *args, phase_dim: int = 4, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.phase_dim = int(phase_dim)
        shape = (self.buffer_size, self.n_envs)
        self.phase_targets = np.zeros((*shape, self.phase_dim), dtype=np.float32)
        self.switch_targets = np.zeros((*shape, 1), dtype=np.float32)
        self.spar_valid = np.zeros((*shape, 1), dtype=np.float32)
        self.phase_labels = np.zeros((*shape, 1), dtype=np.int64)
        self.stall_scores = np.zeros((*shape, 1), dtype=np.float32)
        self.sync_scores = np.zeros((*shape, 1), dtype=np.float32)
        self.align_progresses = np.zeros((*shape, 1), dtype=np.float32)
        self.insert_progresses = np.zeros((*shape, 1), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        pos = self.pos
        for env_idx, info in enumerate(infos):
            phase_target = info.get("spar_phase_target")
            if phase_target is None:
                # No SPAR labels (plain env) -> mark invalid, default to search.
                self.phase_targets[pos, env_idx] = 0.0
                self.phase_targets[pos, env_idx, 0] = 1.0
                self.switch_targets[pos, env_idx, 0] = 0.0
                self.spar_valid[pos, env_idx, 0] = 0.0
                self.phase_labels[pos, env_idx, 0] = 0
                self.stall_scores[pos, env_idx, 0] = 0.0
                self.sync_scores[pos, env_idx, 0] = 0.0
                self.align_progresses[pos, env_idx, 0] = 0.0
                self.insert_progresses[pos, env_idx, 0] = 0.0
                continue

            probs = np.asarray(phase_target, dtype=np.float32).reshape(-1)
            if probs.shape[0] != self.phase_dim:
                raise ValueError(
                    f"Expected spar_phase_target with dim {self.phase_dim}, got {probs.shape[0]}"
                )
            total = float(np.sum(probs))
            if total <= 0.0:
                probs = np.zeros(self.phase_dim, dtype=np.float32)
                probs[0] = 1.0
            else:
                probs = probs / total

            self.phase_targets[pos, env_idx] = probs
            self.switch_targets[pos, env_idx, 0] = float(info.get("spar_switch_target", 0.0))
            self.spar_valid[pos, env_idx, 0] = float(info.get("spar_valid", 1.0))
            self.phase_labels[pos, env_idx, 0] = int(
                info.get("spar_phase_idx", int(np.argmax(probs)))
            )
            self.stall_scores[pos, env_idx, 0] = float(
                np.clip(info.get("spar_stall_score", 0.0), 0.0, 1.0)
            )
            self.sync_scores[pos, env_idx, 0] = float(
                np.clip(info.get("spar_sync_score", 0.0), 0.0, 1.0)
            )
            self.align_progresses[pos, env_idx, 0] = float(
                np.clip(info.get("spar_align_progress", 0.0), 0.0, 1.0)
            )
            self.insert_progresses[pos, env_idx, 0] = float(
                np.clip(info.get("spar_insert_progress", 0.0), 0.0, 1.0)
            )

        super().add(obs, next_obs, action, reward, done, infos)

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
        env_indices: Optional[np.ndarray] = None,
    ) -> SPARReplayBufferSamples:
        if env_indices is None:
            env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(
                self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env
            )
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )
        base = ReplayBufferSamples(*tuple(map(self.to_torch, data)))
        return SPARReplayBufferSamples(
            base.observations,
            base.actions,
            base.next_observations,
            base.dones,
            base.rewards,
            self.to_torch(self.phase_targets[batch_inds, env_indices, :]),
            self.to_torch(self.switch_targets[batch_inds, env_indices, :]),
            self.to_torch(self.spar_valid[batch_inds, env_indices, :]),
            th.as_tensor(self.phase_labels[batch_inds, env_indices, :], device=self.device, dtype=th.long),
            self.to_torch(self.stall_scores[batch_inds, env_indices, :]),
            self.to_torch(self.sync_scores[batch_inds, env_indices, :]),
            self.to_torch(self.align_progresses[batch_inds, env_indices, :]),
            self.to_torch(self.insert_progresses[batch_inds, env_indices, :]),
        )
