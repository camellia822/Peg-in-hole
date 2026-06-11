"""SPAR policy networks (M1: phase awareness + conditioning).

``SPARActor`` adds an independent phase branch that predicts a 4-way soft phase
distribution (search / align / insert / recovery) and a switch logit. The phase
distribution is encoded into a small context vector and concatenated into the
actor latent before ``mu`` / ``log_std`` (conditioning). ``SPARContinuousCritic``
mirrors this by concatenating an encoded phase context into the Q-network input.

A scalar gate ``m1_phase_gate`` (set by the algorithm during warmup/ramp) scales
the phase signal so that at gate=0 the policy is numerically equivalent to plain
SAC (encoder of a zero vector is a learnable constant the heads absorb), and the
phase conditioning fades in monotonically -> clean SAC -> SAC+M1 ladder.
"""

from typing import Dict, List, Optional, Tuple, Type

import torch as th
from torch import nn
from torch.nn import functional as F

from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, create_mlp
from stable_baselines3.sac.policies import Actor, SACPolicy, LOG_STD_MAX, LOG_STD_MIN


class SPARActor(Actor):
    """SAC actor with an auxiliary phase head and phase-conditioned outputs."""

    def __init__(
        self,
        *args,
        phase_dim: int = 4,
        phase_ctx_dim: int = 16,
        phase_hidden: int = 64,
        phase_conditioned: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if self.use_sde and phase_conditioned:
            raise NotImplementedError("SPARActor phase conditioning is not implemented for gSDE.")

        self.phase_dim = int(phase_dim)
        self.phase_ctx_dim = int(phase_ctx_dim)
        self.phase_conditioned = bool(phase_conditioned)
        self.m1_phase_gate: float = 0.0  # updated by the algorithm each train() call

        features_dim = self.features_dim
        latent_dim = self.mu.in_features  # = net_arch[-1]
        action_dim = self.mu.out_features

        # Independent phase branch off the (shared) features extractor output.
        self.phase_branch = nn.Sequential(
            nn.Linear(features_dim, phase_hidden),
            nn.ReLU(),
            nn.Linear(phase_hidden, phase_hidden),
            nn.ReLU(),
        )
        self.phase_head = nn.Linear(phase_hidden, self.phase_dim)
        self.switch_head = nn.Linear(phase_hidden, 1)
        self.phase_encoder = nn.Sequential(
            nn.Linear(self.phase_dim, self.phase_ctx_dim),
            nn.ReLU(),
        )

        if self.phase_conditioned:
            self.mu = nn.Linear(latent_dim + self.phase_ctx_dim, action_dim)
            self.log_std = nn.Linear(latent_dim + self.phase_ctx_dim, action_dim)

    def phase_logits(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """Return (phase_logits [B,4], switch_logit [B,1]) for a batch of states."""
        features = self.extract_features(obs, self.features_extractor)
        hidden = self.phase_branch(features)
        return self.phase_head(hidden), self.switch_head(hidden)

    def phase_probs(self, obs: th.Tensor) -> th.Tensor:
        logits, _ = self.phase_logits(obs)
        return F.softmax(logits, dim=1)

    def get_action_dist_params(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Dict[str, th.Tensor]]:
        features = self.extract_features(obs, self.features_extractor)
        latent_pi = self.latent_pi(features)
        if self.phase_conditioned:
            hidden = self.phase_branch(features)
            probs = F.softmax(self.phase_head(hidden), dim=1)
            ctx = self.phase_encoder(self.m1_phase_gate * probs.detach())
            latent_pi = th.cat([latent_pi, ctx], dim=1)
        mean_actions = self.mu(latent_pi)
        log_std = self.log_std(latent_pi)
        log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean_actions, log_std, {}


class SPARContinuousCritic(ContinuousCritic):
    """Continuous critic whose Q-networks are conditioned on an encoded phase."""

    def __init__(
        self,
        observation_space,
        action_space,
        net_arch: List[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
        phase_dim: int = 4,
        phase_ctx_dim: int = 16,
        phase_conditioned: bool = True,
    ) -> None:
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
        self.phase_dim = int(phase_dim)
        self.phase_ctx_dim = int(phase_ctx_dim)
        self.phase_conditioned = bool(phase_conditioned)

        self.phase_encoder = nn.Sequential(
            nn.Linear(self.phase_dim, self.phase_ctx_dim),
            nn.ReLU(),
        )

        if self.phase_conditioned:
            action_dim = get_action_dim(action_space)
            self.q_networks = []
            for idx in range(n_critics):
                q_net = nn.Sequential(
                    *create_mlp(features_dim + action_dim + self.phase_ctx_dim, 1, net_arch, activation_fn)
                )
                self.add_module(f"qf{idx}", q_net)
                self.q_networks.append(q_net)

    def _encode_phase(self, phase_context: Optional[th.Tensor], batch: int, device) -> th.Tensor:
        if phase_context is None:
            zeros = th.zeros(batch, self.phase_dim, device=device)
            return self.phase_encoder(zeros)
        return self.phase_encoder(phase_context)

    def forward(
        self, obs: th.Tensor, actions: th.Tensor, phase_context: Optional[th.Tensor] = None
    ) -> Tuple[th.Tensor, ...]:
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs, self.features_extractor)
        if self.phase_conditioned:
            ctx = self._encode_phase(phase_context, features.shape[0], features.device)
            qvalue_input = th.cat([features, actions, ctx], dim=1)
        else:
            qvalue_input = th.cat([features, actions], dim=1)
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)


class SPARPolicy(SACPolicy):
    """SAC policy that builds ``SPARActor`` / ``SPARContinuousCritic``."""

    def __init__(
        self,
        *args,
        phase_dim: int = 4,
        phase_ctx_dim: int = 16,
        phase_hidden: int = 64,
        phase_conditioned: bool = True,
        **kwargs,
    ) -> None:
        self._spar_phase_dim = int(phase_dim)
        self._spar_phase_ctx_dim = int(phase_ctx_dim)
        self._spar_phase_hidden = int(phase_hidden)
        self._spar_phase_conditioned = bool(phase_conditioned)
        super().__init__(*args, **kwargs)

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> SPARActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return SPARActor(
            phase_dim=self._spar_phase_dim,
            phase_ctx_dim=self._spar_phase_ctx_dim,
            phase_hidden=self._spar_phase_hidden,
            phase_conditioned=self._spar_phase_conditioned,
            **actor_kwargs,
        ).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> SPARContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return SPARContinuousCritic(
            phase_dim=self._spar_phase_dim,
            phase_ctx_dim=self._spar_phase_ctx_dim,
            phase_conditioned=self._spar_phase_conditioned,
            **critic_kwargs,
        ).to(self.device)
