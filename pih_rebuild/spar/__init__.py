"""SPAR-SAC modules (M1 phase-awareness, M2 phase-adaptive entropy).

Built directly on stable_baselines3.SAC for the redesigned UR5 dual-peg env.
The env writes auxiliary phase labels into ``info`` (see
``UR5DualPegEnv._spar_labels``); these modules consume them without touching
observation / reward / action / physics, so plain SAC stays comparable.
"""

from pih_rebuild.spar.buffers import SPARReplayBuffer, SPARReplayBufferSamples
from pih_rebuild.spar.policies import SPARActor, SPARContinuousCritic, SPARPolicy
from pih_rebuild.spar.algorithm import SPARSAC

__all__ = [
    "SPARReplayBuffer",
    "SPARReplayBufferSamples",
    "SPARActor",
    "SPARContinuousCritic",
    "SPARPolicy",
    "SPARSAC",
]
