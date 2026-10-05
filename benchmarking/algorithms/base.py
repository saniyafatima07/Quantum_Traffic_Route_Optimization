import dataclasses
from typing import List, Dict, Any, Optional
from benchmark.datasets.problem import Problem

@dataclasses.dataclass
class Result:
    route: List[List[int]]  # List of routes for each vehicle
    cost: float
    distance: float
    travel_time: float
    congestion: float
    iterations: int
    runtime: float
    history: List[float]  # Objective value at each iteration (for convergence plots)
    feasible: bool

class Algorithm:
    def __init__(self, name: str):
        self.name = name

    def solve(self, problem: Problem, traffic_sim=None, seed: int = 42, **kwargs) -> Result:
        """
        Solve the given routing problem.
        
        Args:
            problem: The dataset/problem instance.
            traffic_sim: Optional traffic simulator to calculate dynamic travel times.
            seed: Random seed for reproducibility.
            **kwargs: Algorithm specific hyperparameters (e.g., pop_size, max_iter)
            
        Returns:
            Result object containing the best found routes, metrics, and convergence history.
        """
        raise NotImplementedError("Subclasses must implement the solve method.")
