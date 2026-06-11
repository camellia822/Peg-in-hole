"""SPAR-SAC algorithm: M1 (phase awareness + conditioning) and M2 (phase-adaptive
entropy), built directly on stable_baselines3.SAC.

Ladder (perturb 0.65, seed 7, 200k):
    SAC            : plain SB3 SAC (separate run)
    SAC+M1         : SPARSAC(enable_m2=False)
    SAC+M1+M2      : SPARSAC(enable_m2=True)

M1 -- the actor predicts a soft 4-phase distribution (search/align/insert/
recovery) supervised by the env's auxiliary labels (soft CE + switch BCE +
temporal smoothness). The predicted phase is encoded and fed (gated) into the
actor and critic, so the policy/value functions become phase-conditioned. A
warmup/ramp gate makes SAC -> SAC+M1 a smooth, monotone addition.

M2 -- replaces the single fixed target entropy with a per-sample, phase-adaptive
target: exploration-heavy phases (search, recovery) demand more entropy, while
precision phases (align, insert) demand less, with continuous modulation from
stall / sync / progress signals. Blended in over a warmup/ramp tuned for 200k.
"""

from typing import Optional

import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update

from pih_rebuild.spar.buffers import SPARReplayBuffer
from pih_rebuild.spar.policies import SPARPolicy


def _ramp(step: int, warmup: int, ramp: int) -> float:
    if ramp <= 0:
        return 1.0 if step >= warmup else 0.0
    return float(np.clip((step - warmup) / ramp, 0.0, 1.0))


class SPARSAC(SAC):
    """SAC + M1 (phase conditioning) + optional M2 (phase-adaptive entropy)."""

    def __init__(
        self,
        policy="MlpPolicy",
        env=None,
        *,
        enable_m2: bool = True,
        phase_dim: int = 4,
        phase_ctx_dim: int = 16,
        phase_hidden: int = 64,
        # M1 conditioning gate schedule
        m1_warmup_steps: int = 20_000,
        m1_ramp_steps: int = 30_000,
        # M1 auxiliary loss coefficients
        phase_loss_coef: float = 0.05,
        switch_loss_coef: float = 0.02,
        phase_smooth_coef: float = 0.01,
        # M2 dynamic-entropy schedule
        m2_warmup_steps: int = 20_000,
        m2_ramp_steps: int = 30_000,
        # M2 per-phase entropy scales (search, align, insert, recovery)
        entropy_phase_scales=(1.05, 0.90, 0.85, 1.02),
        stall_coef: float = 0.05,
        sync_coef: float = 0.05,
        align_progress_coef: float = 0.04,
        insert_progress_coef: float = 0.06,
        m2_min_scale: float = 0.6,
        m2_max_scale: float = 1.5,
        policy_kwargs: Optional[dict] = None,
        replay_buffer_class=None,
        replay_buffer_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> None:
        self.enable_m2 = bool(enable_m2)
        self.phase_dim = int(phase_dim)
        self.m1_warmup_steps = int(m1_warmup_steps)
        self.m1_ramp_steps = int(m1_ramp_steps)
        self.phase_loss_coef = float(phase_loss_coef)
        self.switch_loss_coef = float(switch_loss_coef)
        self.phase_smooth_coef = float(phase_smooth_coef)
        self.m2_warmup_steps = int(m2_warmup_steps)
        self.m2_ramp_steps = int(m2_ramp_steps)
        self.entropy_phase_scales = tuple(float(s) for s in entropy_phase_scales)
        if len(self.entropy_phase_scales) != self.phase_dim:
            raise ValueError("entropy_phase_scales length must equal phase_dim")
        self.stall_coef = float(stall_coef)
        self.sync_coef = float(sync_coef)
        self.align_progress_coef = float(align_progress_coef)
        self.insert_progress_coef = float(insert_progress_coef)
        self.m2_min_scale = float(m2_min_scale)
        self.m2_max_scale = float(m2_max_scale)

        policy_kwargs = dict(policy_kwargs or {})
        policy_kwargs.update(
            phase_dim=phase_dim,
            phase_ctx_dim=phase_ctx_dim,
            phase_hidden=phase_hidden,
            phase_conditioned=True,
        )
        if replay_buffer_class is None:
            replay_buffer_class = SPARReplayBuffer
        replay_buffer_kwargs = dict(replay_buffer_kwargs or {})
        replay_buffer_kwargs.setdefault("phase_dim", phase_dim)

        super().__init__(
            SPARPolicy,
            env,
            policy_kwargs=policy_kwargs,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            **kwargs,
        )

    def _setup_model(self) -> None:
        super()._setup_model()
        self._phase_scale = th.tensor(
            self.entropy_phase_scales, device=self.device, dtype=th.float32
        ).reshape(1, -1)

    def _dynamic_target_entropy(self, data, beta: float) -> th.Tensor:
        """Per-sample phase-adaptive target entropy (B,1); falls back to the base
        target entropy where phase labels are invalid or M2 is disabled."""
        base = float(self.target_entropy)
        if not self.enable_m2 or beta <= 0.0:
            return th.full((data.phase_targets.shape[0], 1), base, device=self.device)

        scale = (data.phase_targets * self._phase_scale).sum(dim=1, keepdim=True)
        scale = scale + self.stall_coef * data.stall_scores + self.sync_coef * data.sync_scores
        scale = scale - self.align_progress_coef * data.align_progresses
        scale = scale - self.insert_progress_coef * data.insert_progresses
        scale = th.clamp(scale, self.m2_min_scale, self.m2_max_scale)

        dyn = base * scale  # base is negative -> larger scale => more entropy demanded
        eff = (1.0 - beta) * base + beta * dyn
        # only apply where the env provided valid labels
        eff = data.spar_valid * eff + (1.0 - data.spar_valid) * base
        return eff

    def _phase_contexts(self, data, gate: float):
        """Return (ctx_obs, ctx_next) detached phase contexts for the critic,
        teacher-forced from env labels early and from predictions later."""
        with th.no_grad():
            pred_obs = self.actor.phase_probs(data.observations)
            pred_next = self.actor.phase_probs(data.next_observations)
        if gate <= 0.0:
            return None, None
        tf = 1.0 - gate  # teacher forcing weight (env labels early)
        env_next = data.spar_valid * data.phase_targets + (1.0 - data.spar_valid) * pred_next
        blend_next = tf * env_next + (1.0 - tf) * pred_next
        ctx_obs = gate * pred_obs
        ctx_next = gate * blend_next
        return ctx_obs, ctx_next

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        phase_ce_losses, switch_losses, smooth_losses = [], [], []
        dyn_targets = []

        step = self.num_timesteps
        gate = _ramp(step, self.m1_warmup_steps, self.m1_ramp_steps)
        beta = _ramp(step, self.m2_warmup_steps, self.m2_ramp_steps) if self.enable_m2 else 0.0
        self.actor.m1_phase_gate = gate

        for _ in range(gradient_steps):
            data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            valid_count = data.spar_valid.sum().clamp(min=1.0)

            if self.use_sde:
                self.actor.reset_noise()

            ctx_obs, ctx_next = self._phase_contexts(data, gate)

            actions_pi, log_prob = self.actor.action_log_prob(data.observations)
            log_prob = log_prob.reshape(-1, 1)

            # ---- M2: phase-adaptive target entropy + entropy coefficient ----
            eff_target = self._dynamic_target_entropy(data, beta)
            dyn_targets.append(float(eff_target.mean().item()))

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + eff_target).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor
            ent_coefs.append(ent_coef.item())

            if ent_coef_loss is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            # ---- critic update (phase-conditioned) ----
            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(data.next_observations)
                next_q_values = th.cat(
                    self.critic_target(data.next_observations, next_actions, phase_context=ctx_next), dim=1
                )
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = data.rewards + (1 - data.dones) * self.gamma * next_q_values

            current_q_values = self.critic(data.observations, data.actions, phase_context=ctx_obs)
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # ---- actor update (SAC loss + M1 auxiliary losses) ----
            q_values_pi = th.cat(
                self.critic(data.observations, actions_pi, phase_context=ctx_obs), dim=1
            )
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()

            logits_obs, _ = self.actor.phase_logits(data.observations)
            logits_next, switch_next = self.actor.phase_logits(data.next_observations)

            log_probs_next = F.log_softmax(logits_next, dim=1)
            ce = -(data.phase_targets * log_probs_next).sum(dim=1, keepdim=True)
            phase_ce = (ce * data.spar_valid).sum() / valid_count

            bce = F.binary_cross_entropy_with_logits(switch_next, data.switch_targets, reduction="none")
            switch_loss = (bce * data.spar_valid).sum() / valid_count

            probs_obs = F.softmax(logits_obs, dim=1)
            probs_next = F.softmax(logits_next, dim=1)
            smooth = ((probs_obs - probs_next) ** 2).sum(dim=1, keepdim=True) * (1.0 - data.switch_targets)
            smooth_loss = (smooth * data.spar_valid).sum() / valid_count

            actor_loss = (
                actor_loss
                + self.phase_loss_coef * phase_ce
                + self.switch_loss_coef * switch_loss
                + self.phase_smooth_coef * smooth_loss
            )
            actor_losses.append(float(actor_loss.item()))
            phase_ce_losses.append(float(phase_ce.item()))
            switch_losses.append(float(switch_loss.item()))
            smooth_losses.append(float(smooth_loss.item()))

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            if self._n_updates % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)
            self._n_updates += 1

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if ent_coef_losses:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
        self.logger.record("spar/m1_gate", gate)
        self.logger.record("spar/m2_beta", beta)
        self.logger.record("spar/phase_ce", float(np.mean(phase_ce_losses)))
        self.logger.record("spar/switch_loss", float(np.mean(switch_losses)))
        self.logger.record("spar/smooth_loss", float(np.mean(smooth_losses)))
        self.logger.record("spar/target_entropy", float(np.mean(dyn_targets)))
