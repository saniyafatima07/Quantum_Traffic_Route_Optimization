"""Classical baselines: the control group the quantum methods must beat."""

from __future__ import annotations

from .construction import SavingsBaseline
from .ga import GA
from .local_search import LocalSearchRefiner
from .pso import PSO
from .sa import SimulatedAnnealing

__all__ = [
    "GA",
    "LocalSearchRefiner",
    "PSO",
    "SavingsBaseline",
    "SimulatedAnnealing",
]
