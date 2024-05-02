"""Emitters output new candidate solutions in QD algorithms.

All emitters should inherit from :class:`EmitterBase`, except for emitters
designed for differentiable quality diversity (DQD), which should instead
inherit from :class:`DQDEmitterBase`.

.. note::
    Emitters provided here take on the data type of the archive passed to their
    constructor. For instance, if an archive has dtype ``np.float64``, then an
    emitter created with that archive will emit solutions with dtype
    ``np.float64``.

.. autosummary::
    :toctree:

    ribs2.emitters.EvolutionStrategyEmitter
    ribs2.emitters.GradientAborescenceEmitter
    ribs2.emitters.GaussianEmitter
    ribs2.emitters.IsoLineEmitter
    ribs2.emitters.EmitterBase
    ribs2.emitters.DQDEmitterBase
"""
from ribs2.emitters._dqd_emitter_base import DQDEmitterBase
from ribs2.emitters._emitter_base import EmitterBase
from ribs2.emitters._evolution_strategy_emitter import EvolutionStrategyEmitter
from ribs2.emitters._gaussian_emitter import GaussianEmitter
from ribs2.emitters._gradient_aborescence_emitter import \
  GradientAborescenceEmitter
from ribs2.emitters._proximal_policy_gradient_arborescence_emitter import PPGAEmitter
from ribs2.emitters._iso_line_emitter import IsoLineEmitter

__all__ = [
    "EvolutionStrategyEmitter",
    "GradientAborescenceEmitter",
    "PPGAEmitter",
    "GaussianEmitter",
    "IsoLineEmitter",
    "EmitterBase",
    "DQDEmitterBase",
]
