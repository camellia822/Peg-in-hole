"""SPAR-SAC M1: soft phase-aware network components.

``PhaseNet`` is the auxiliary branch attached on top of the policy's shared
feature layer. It predicts a four-class phase distribution and a scalar
phase-boundary switch probability. The same module is consulted by both the
actor and the critic so that the conditioning signal is consistent; it is
owned (and optimised) by the actor through the auxiliary loss, while the critic
reads it with a detached graph.
"""

from __future__ import annotations

import torch as th
import torch.nn as nn


class PhaseNet(nn.Module):
    """Phase + switch prediction head sitting on the shared observation features.

    :param obs_dim: dimensionality of the (flattened) observation.
    :param hidden: width of the two-layer prediction trunk.
    :param n_phases: number of assembly phases (search/align/insert/recovery).
    """

    def __init__(self, obs_dim: int, hidden: int = 64, n_phases: int = 4):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.n_phases = int(n_phases)
        self.trunk = nn.Sequential(
            nn.Linear(self.obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.phase_head = nn.Linear(hidden, self.n_phases)
        self.switch_head = nn.Linear(hidden, 1)

    def forward(self, obs: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """Return ``(phase_logits, switch_logit)``."""
        h = self.trunk(obs)
        return self.phase_head(h), self.switch_head(h)

    def phase_probs(self, obs: th.Tensor) -> th.Tensor:
        """Softmax phase distribution of shape ``(batch, n_phases)``."""
        logits, _ = self.forward(obs)
        return th.softmax(logits, dim=-1)
