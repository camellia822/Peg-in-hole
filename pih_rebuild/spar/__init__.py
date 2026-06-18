"""SPAR-SAC modules for the rebuilt UR5 dual-peg task.

The package keeps its top-level import lightweight (only the numpy-only label
generator) so that ``ur5_dual_peg_env`` can import :class:`SparLabeler` without
pulling in torch / stable-baselines3. The heavy components are imported from
their submodules directly by the training entry point::

    from pih_rebuild.spar.algorithm import SPARSAC
    from pih_rebuild.spar.policies import SPARPolicy
    from pih_rebuild.spar.buffers import SPARReplayBuffer
"""

from pih_rebuild.spar.labels import PHASE_NAMES, SparLabeler

__all__ = ["SparLabeler", "PHASE_NAMES"]
