"""SPAR-SAC: Soft Phase-Aware Reinforcement learning SAC.

A drop-in SAC subclass that composes the three patent modules on top of the
rebuilt UR5 dual-peg task. The modules are orthogonal and individually
switchable so they can be ablated as ``SAC < +M1 < +M1+M2 < +M1+M2+M3``:

* **M1 (soft phase-aware):** a shared ``PhaseNet`` predicts a soft phase
  distribution that conditions the actor and critic; trained by an auxiliary
  loss (soft cross-entropy + switch BCE + temporal smoothness). A ``gate`` ramp
  fades the module in from a pure-SAC start.
* **M2 (dynamic target entropy):** the fixed SAC target entropy is replaced by a
  per-sample value computed from the phase distribution (high exploration in
  search, fine control in insert, mild exploration in recovery) plus progress /
  anomaly modulation. Only the temperature loss is affected.
* **M3 (asymmetric prioritised replay):** the critic and actor sample two
  separate prioritised mini-batches (critic uses the full TD magnitude, actor a
  clipped one) with importance-sampling correction.

With every module disabled the ``train`` loop is numerically identical to
vanilla SAC, which keeps the ablation honest.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch as th
from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update
from torch.nn import functional as F


class SPARSAC(SAC):
    def __init__(
        self,
        policy,
        env,
        *,
        enable_m1: bool = True,
        enable_m2: bool = False,
        enable_m3: bool = False,
        m1_warmup: int = 20000,
        m1_ramp: int = 30000,
        m2_warmup: int = 20000,
        m2_ramp: int = 30000,
        m1_phase_coef: float = 0.05,
        m1_switch_coef: float = 0.02,
        m1_smooth_coef: float = 0.01,
        m2_phase_scale: Tuple[float, float, float, float] = (0.85, 0.95, 1.15, 1.0),
        m2_stall_coef: float = 0.06,
        m2_sync_coef: float = 0.05,
        m2_align_coef: float = 0.05,
        m2_insert_coef: float = 0.04,
        m2_scale_min: float = 0.6,
        m2_scale_max: float = 1.5,
        m3_beta_start: float = 0.4,
        **kwargs,
    ):
        self.enable_m1 = bool(enable_m1)
        self.enable_m2 = bool(enable_m2)
        self.enable_m3 = bool(enable_m3)
        self.m1_warmup = int(m1_warmup)
        self.m1_ramp = int(m1_ramp)
        self.m2_warmup = int(m2_warmup)
        self.m2_ramp = int(m2_ramp)
        self.m1_phase_coef = float(m1_phase_coef)
        self.m1_switch_coef = float(m1_switch_coef)
        self.m1_smooth_coef = float(m1_smooth_coef)
        self._m2_phase_scale_tuple = tuple(float(x) for x in m2_phase_scale)
        self.m2_stall_coef = float(m2_stall_coef)
        self.m2_sync_coef = float(m2_sync_coef)
        self.m2_align_coef = float(m2_align_coef)
        self.m2_insert_coef = float(m2_insert_coef)
        self.m2_scale_min = float(m2_scale_min)
        self.m2_scale_max = float(m2_scale_max)
        self.m3_beta_start = float(m3_beta_start)
        self.m2_beta = 0.0
        super().__init__(policy, env, **kwargs)
        self._phase_scale = th.tensor(
            self._m2_phase_scale_tuple, device=self.device, dtype=th.float32
        ).reshape(1, -1)

    @staticmethod
    def _ramp(step: int, warmup: int, ramp: int) -> float:
        if step < warmup:
            return 0.0
        if ramp <= 0:
            return 1.0
        return float(min(1.0, (step - warmup) / ramp))

    def _dynamic_target_entropy(self, data) -> th.Tensor:
        """Per-sample target entropy (patent eq. 4-5)."""
        base = float(self.target_entropy)  # negative scalar, e.g. -3
        scale = (data.phase_target * self._phase_scale).sum(dim=1, keepdim=True)
        # Stalling / desync raise the entropy target (less negative); alignment /
        # insertion progress lower it (more negative) for fine control.
        scale = scale - self.m2_stall_coef * data.stall - self.m2_sync_coef * data.sync_gap
        scale = scale + self.m2_align_coef * data.align_progress + self.m2_insert_coef * data.insert_progress
        scale = th.clamp(scale, self.m2_scale_min, self.m2_scale_max)
        dyn = base * scale
        return (1.0 - self.m2_beta) * base + self.m2_beta * dyn

    def _phase_aux_loss(self, data) -> th.Tensor:
        """M1 auxiliary loss: soft CE + switch BCE + temporal smoothness."""
        logits, switch_logit = self.actor.predict_phase(data.observations)
        log_probs = th.log_softmax(logits, dim=1)
        ce = -(data.phase_target * log_probs).sum(dim=1).mean()
        bce = F.binary_cross_entropy_with_logits(switch_logit, data.switch_target)
        probs = th.softmax(logits, dim=1)
        with th.no_grad():
            next_logits, _ = self.actor.predict_phase(data.next_observations)
            next_probs = th.softmax(next_logits, dim=1)
        smooth = ((probs - next_probs) ** 2).sum(dim=1, keepdim=True)
        smooth = ((1.0 - data.switch_target) * smooth).mean()
        return self.m1_phase_coef * ce + self.m1_switch_coef * bce + self.m1_smooth_coef * smooth

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        step = self.num_timesteps
        gate = self._ramp(step, self.m1_warmup, self.m1_ramp) if self.enable_m1 else 0.0
        self.m2_beta = self._ramp(step, self.m2_warmup, self.m2_ramp) if self.enable_m2 else 0.0
        self.policy.set_spar_gate(gate)
        if self.enable_m3:
            frac = min(1.0, step / max(int(self._total_timesteps), 1))
            self.replay_buffer.beta_is = self.m3_beta_start + frac * (1.0 - self.m3_beta_start)

        ent_coef_losses, ent_coefs, target_entropies = [], [], []
        actor_losses, critic_losses, aux_losses = [], [], []

        for gradient_step in range(gradient_steps):
            if self.enable_m3:
                critic_data = self.replay_buffer.sample_prioritized(batch_size, "critic", env=self._vec_normalize_env)
                actor_data = self.replay_buffer.sample_prioritized(batch_size, "actor", env=self._vec_normalize_env)
            else:
                critic_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
                actor_data = critic_data

            # ---- entropy temperature (M2 dynamic target) ----
            actions_pi, log_prob = self.actor.action_log_prob(actor_data.observations)
            log_prob = log_prob.reshape(-1, 1)
            if self.enable_m2:
                target_ent = self._dynamic_target_entropy(actor_data)
                target_entropies.append(float(target_ent.mean().item()))
            else:
                target_ent = self.target_entropy
                target_entropies.append(float(self.target_entropy))

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + target_ent).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor
            ent_coefs.append(ent_coef.item())
            if ent_coef_loss is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            # ---- critic update (M3 prioritised, IS-weighted) ----
            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(critic_data.next_observations)
                next_q = th.cat(self.critic_target(critic_data.next_observations, next_actions), dim=1)
                next_q, _ = th.min(next_q, dim=1, keepdim=True)
                next_q = next_q - ent_coef * next_log_prob.reshape(-1, 1)
                target_q = critic_data.rewards + (1 - critic_data.dones) * self.gamma * next_q
            current_q = self.critic(critic_data.observations, critic_data.actions)
            is_w_c = critic_data.is_weights
            critic_loss = 0.5 * sum((is_w_c * (cq - target_q) ** 2).mean() for cq in current_q)
            critic_losses.append(critic_loss.item())
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            if self.enable_m3:
                with th.no_grad():
                    q_mean = th.cat(current_q, dim=1).mean(dim=1, keepdim=True)
                    td = (target_q - q_mean).abs().reshape(-1).cpu().numpy()
                self.replay_buffer.update_priorities(critic_data.batch_inds, critic_data.env_indices, td)

            # ---- actor update (M3 actor stream + M1 auxiliary loss) ----
            q_pi = th.cat(self.critic(actor_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_pi, dim=1, keepdim=True)
            is_w_a = actor_data.is_weights
            actor_loss = (is_w_a * (ent_coef * log_prob - min_qf_pi)).mean()
            actor_losses.append(actor_loss.item())
            if self.enable_m1 and gate > 0.0:
                aux = self._phase_aux_loss(actor_data)
                actor_loss = actor_loss + gate * aux
                aux_losses.append(float(aux.item()))
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
        self.logger.record("spar/gate", gate)
        self.logger.record("spar/m2_beta", self.m2_beta)
        self.logger.record("spar/target_entropy", float(np.mean(target_entropies)))
        if len(aux_losses) > 0:
            self.logger.record("spar/phase_aux_loss", float(np.mean(aux_losses)))
        if self.enable_m3:
            self.logger.record("spar/is_beta", float(self.replay_buffer.beta_is))
