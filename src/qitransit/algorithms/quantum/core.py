"""Quantum-inspired primitives shared by the QPSO and QGA implementations.

Nothing here touches a quantum device.  The value of these routines is that the
*mean-field* update and the *quantum fluctuation* term are what give
quantum-inspired swarm search its balance between exploration and exploitation,
and modelling them explicitly makes that mechanism inspectable.

The state a particle carries is a Q-bit amplitude ``theta``; the real-valued
position the problem can see is produced by measuring it.  Two consequences
matter for the search:

* ``sin^2(theta)`` is invariant under ``theta -> theta + pi``, so the collapse
  step may add a full half-turn of phase without the measured position jumping —
  the particle explores without the objective seeing noise.
* re-encoding a measured position with ``arcsin(sqrt(x))`` returns the particle
  to a coherent superposition instead of freezing it at a classical value, so
  the next iteration starts from a distribution rather than a point.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12
HALF_TURN = float(np.pi)


def measure_position(theta: np.ndarray) -> np.ndarray:
    """Collapse a Q-bit amplitude to the real axis.

    The measurement basis of a Q-bit has ``|0>`` and ``|1>`` as eigenstates, so
    the measured amplitude is ``sin^2(theta)``: exactly ``0`` for ``|0>`` and
    exactly ``1`` for ``|1>``.
    """
    return np.sin(theta) ** 2


def amplitude_from_position(x: np.ndarray) -> np.ndarray:
    """The Q-bit amplitude a measured position corresponds to.

    Inverts :func:`measure_position` on the principal branch, so a particle
    re-enters the update as a coherent superposition rather than a collapsed
    classical value.
    """
    return np.arcsin(np.sqrt(np.clip(x, 0.0, 1.0)))


def half_turn_phase(theta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A random phase in ``[0, pi)``, i.e. a flip about ``|0>`` with no net effect
    on the measured position for a fixed amplitude."""
    return theta + HALF_TURN * rng.random(np.shape(theta))


def quantum_fluctuation(
    x: np.ndarray,
    mean_pbest: np.ndarray,
    iteration: int,
    spread: float = 0.5,
) -> np.ndarray:
    """The per-dimension collapse probability of the QPSO update.

    ``w = exp(-|t| |x - mean(pbest)|)`` shrinks as the swarm converges and as
    the individual particle strays from the swarm, so late iterations explore
    less and early iterations explore more.  ``spread`` widens the band, trading
    convergence speed for diversity.
    """
    deviation = np.abs(x - mean_pbest)
    return np.exp(-(iteration + 1.0) * deviation / (2.0 * max(spread, EPS)))


def mean_field(
    pbest: np.ndarray,
    gbest: np.ndarray,
    alpha: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """The QPSO attractor: the global best directly, or a convex blend.

    With probability ``alpha`` the mean field collapses straight onto the global
    best, which exploits; otherwise it interpolates between the particle's own
    best and the global best, which explores.
    """
    shape = np.shape(pbest)
    r1 = rng.random(shape)
    r2 = rng.random(shape)
    blend = r1 * pbest + (1.0 - r1) * r2 * gbest
    return np.where(rng.random(shape) < alpha, gbest, blend)


def diversity(bits: np.ndarray) -> float:
    """Fraction of disagreeing columns; zero means the swarm has collapsed."""
    if bits.size == 0:
        return 0.0
    ones = bits.mean(axis=0)
    return float(np.mean(np.minimum(ones, 1.0 - ones)) * 2.0)


def bits_of(theta: np.ndarray) -> np.ndarray:
    """The measured basis state of each amplitude, for diversity accounting."""
    return (measure_position(theta) > 0.5).astype(np.float64)


def normalise(values: np.ndarray) -> np.ndarray:
    """Scale into ``[0, 1]``, guarding the degenerate constant case."""
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi - lo < EPS:
        return np.zeros_like(values, dtype=np.float64)
    return (values - lo) / (hi - lo)
