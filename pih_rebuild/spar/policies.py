"""SPAR-SAC M1: phase-conditioned actor / critic and the owning policy.

The actor and critic keep their standard SB3 structure but receive an extra
phase-conditioning vector. A shared :class:`PhaseNet` predicts the phase
distribution from the (flattened) observation; the distribution is encoded by a
small per-network encoder and concatenated -- scaled by a ramp ``gate`` -- into
the actor's latent and the critic's Q-input.

Ownership / gradient routing is deliberate:

* the ``PhaseNet`` is registered as a *submodule of the actor*, so it is only
  updated by the actor optimiser via the M1 auxiliary loss;
* the actor's RL policy-gradient path detaches the phase probabilities, so the
  reward signal never corrupts the phase representation;
* the critic holds the same ``PhaseNet`` through a non-registered reference and
  always reads it detached, so the critic loss neither updates it nor double
  counts it.

With ``spar_gate == 0`` the conditioning contributes an all-zero vector, which
(because ``mu``/``log_std`` and the Q-heads are linear in their input) makes the
networks numerically equivalent to vanilla SAC -- this is what lets the gate be
ramped up smoothly from a pure-SAC start.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Type

import torch as th
import torch.nn as nn
from gym import spaces
from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.sac.policies import LOG_STD_MAX, LOG_STD_MIN, Actor, SACPolicy

from pih_rebuild.spar.networks import PhaseNet


class SPARActor(Actor):
    """Actor whose action head is conditioned on the predicted phase."""

    def __init__(self, *args, phase_net: PhaseNet, enc_dim: int = 16, n_phases: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        assert not self.use_sde, "SPARActor does not support gSDE."
        # PhaseNet is a *submodule of the actor*: optimised by the actor's
        # optimiser through the auxiliary loss only.
        self.phase_net = phase_net
        self.enc_dim = int(enc_dim)
        self.n_phases = int(n_phases)
        self.spar_gate = 1.0
        self.phase_encoder = nn.Sequential(nn.Linear(self.n_phases, self.enc_dim), nn.ReLU())

        last_layer_dim = self.net_arch[-1] if len(self.net_arch) > 0 else self.features_dim
        action_dim = get_action_dim(self.action_space)
        # Rebuild the action heads to consume the phase-conditioning vector.
        self.mu = nn.Linear(last_layer_dim + self.enc_dim, action_dim)
        self.log_std = nn.Linear(last_layer_dim + self.enc_dim, action_dim)

    def _phase_conditioning(self, features: th.Tensor) -> th.Tensor:
        logits, _ = self.phase_net(features)
        probs = th.softmax(logits, dim=-1).detach()
        return self.spar_gate * self.phase_encoder(probs)

    def get_action_dist_params(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Dict[str, th.Tensor]]:
        features = self.extract_features(obs, self.features_extractor)
        latent_pi = self.latent_pi(features)
        cond = self._phase_conditioning(features)
        latent = th.cat([latent_pi, cond], dim=1)
        mean_actions = self.mu(latent)
        log_std = th.clamp(self.log_std(latent), LOG_STD_MIN, LOG_STD_MAX)
        return mean_actions, log_std, {}

    def predict_phase(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """Phase logits and switch logit, with gradients (for the M1 loss)."""
        features = self.extract_features(obs, self.features_extractor)
        return self.phase_net(features)


class SPARCritic(ContinuousCritic):
    """Continuous critic whose Q-input is conditioned on the predicted phase."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
        phase_net: Optional[PhaseNet] = None,
        enc_dim: int = 16,
        n_phases: int = 4,
    ):
        super().__init__(
            observation_space,
            action_space,
            net_arch,
            features_extractor,
            features_dim,
            activation_fn,
            normalize_images,
            n_critics,
            share_features_extractor,
        )
        # Reference (not a registered submodule) so the critic optimiser never
        # touches the PhaseNet; always read detached.
        self._phase_net_ref = [phase_net]
        self.enc_dim = int(enc_dim)
        self.n_phases = int(n_phases)
        self.spar_gate = 1.0
        self.phase_encoder = nn.Sequential(nn.Linear(self.n_phases, self.enc_dim), nn.ReLU())

        action_dim = get_action_dim(action_space)
        self.q_networks = []
        for idx in range(n_critics):
            q_net = nn.Sequential(
                *create_mlp(features_dim + action_dim + self.enc_dim, 1, net_arch, activation_fn)
            )
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def _phase_conditioning(self, features: th.Tensor) -> th.Tensor:
        phase_net = self._phase_net_ref[0]
        logits, _ = phase_net(features)
        probs = th.softmax(logits, dim=-1).detach()
        return self.spar_gate * self.phase_encoder(probs)

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, ...]:
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs, self.features_extractor)
        cond = self._phase_conditioning(features)
        qvalue_input = th.cat([features, actions, cond], dim=1)
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)

    def q1_forward(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        with th.no_grad():
            features = self.extract_features(obs, self.features_extractor)
        cond = self._phase_conditioning(features)
        return self.q_networks[0](th.cat([features, actions, cond], dim=1))


class SPARPolicy(SACPolicy):
    """SAC policy that wires a shared :class:`PhaseNet` into actor and critic."""

    def __init__(self, *args, phase_enc_dim: int = 16, phase_hidden: int = 64, n_phases: int = 4, **kwargs):
        self._spar_enc_dim = int(phase_enc_dim)
        self._spar_hidden = int(phase_hidden)
        self._spar_n_phases = int(n_phases)
        self.phase_net: Optional[PhaseNet] = None
        super().__init__(*args, **kwargs)

    def _ensure_phase_net(self, obs_dim: int) -> PhaseNet:
        if self.phase_net is None:
            self.phase_net = PhaseNet(
                obs_dim, hidden=self._spar_hidden, n_phases=self._spar_n_phases
            ).to(self.device)
        return self.phase_net

    def make_actor(self, features_extractor: Optional[nn.Module] = None) -> SPARActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        phase_net = self._ensure_phase_net(actor_kwargs["features_dim"])
        return SPARActor(
            **actor_kwargs,
            phase_net=phase_net,
            enc_dim=self._spar_enc_dim,
            n_phases=self._spar_n_phases,
        ).to(self.device)

    def make_critic(self, features_extractor: Optional[nn.Module] = None) -> SPARCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        phase_net = self._ensure_phase_net(critic_kwargs["features_dim"])
        return SPARCritic(
            **critic_kwargs,
            phase_net=phase_net,
            enc_dim=self._spar_enc_dim,
            n_phases=self._spar_n_phases,
        ).to(self.device)

    def set_spar_gate(self, gate: float) -> None:
        gate = float(gate)
        self.actor.spar_gate = gate
        self.critic.spar_gate = gate
        self.critic_target.spar_gate = gate

    def _get_constructor_parameters(self) -> Dict:
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                phase_enc_dim=self._spar_enc_dim,
                phase_hidden=self._spar_hidden,
                n_phases=self._spar_n_phases,
            )
        )
        return data
