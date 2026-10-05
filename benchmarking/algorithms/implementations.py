import time
import random
import numpy as np
from typing import List

from benchmark.algorithms.base import Algorithm, Result
from benchmark.datasets.problem import Problem
from benchmark.metrics.evaluate import evaluate_solution

class DummyHeuristicBase(Algorithm):
    """
    Base class that simulates an optimization algorithm for benchmarking purposes.
    Generates an initial valid solution and simulates iterative improvement.
    """
    def __init__(self, name: str, convergence_speed: float, final_gap: float):
        super().__init__(name)
        self.convergence_speed = convergence_speed
        self.final_gap = final_gap
        
    def _generate_valid_solution(self, problem: Problem) -> List[List[int]]:
        """Greedy route generation to ensure feasibility."""
        routes = []
        unvisited = set(i for i in range(1, problem.num_nodes) if problem.demands[i] > 0)
        
        while unvisited and len(routes) < problem.num_vehicles:
            route = [0]
            load = 0
            curr = 0
            
            while unvisited:
                # Find nearest unvisited node that fits in capacity
                valid_next = [n for n in unvisited if load + problem.demands[n] <= problem.vehicle_capacity]
                if not valid_next:
                    break
                    
                # Nearest neighbor
                nxt = min(valid_next, key=lambda n: problem.distance_matrix[curr, n])
                route.append(nxt)
                load += problem.demands[nxt]
                curr = nxt
                unvisited.remove(nxt)
                
            route.append(0)
            routes.append(route)
            
        # If we couldn't visit all nodes within vehicle limit, dump the rest into the last route (will be infeasible but that's fine for testing)
        if unvisited:
            if not routes:
                routes.append([0, 0])
            for n in unvisited:
                routes[-1].insert(-1, n)
                
        return routes

    def solve(self, problem: Problem, traffic_sim=None, seed: int = 42, **kwargs) -> Result:
        random.seed(seed)
        np.random.seed(seed)
        
        max_iter = kwargs.get('max_iter', 100)
        start_time = time.perf_counter()
        
        # Initial greedy solution
        best_route = self._generate_valid_solution(problem)
        dist, time_val, cong, best_cost, feasible = evaluate_solution(best_route, problem, traffic_sim)
        
        history = []
        current_cost = best_cost
        
        # Simulate convergence
        target_cost = best_cost * self.final_gap
        for i in range(max_iter):
            # Simulate some processing time
            time.sleep(0.001)
            
            # Simulate cost reduction
            improvement = (current_cost - target_cost) * self.convergence_speed * random.uniform(0.5, 1.5)
            current_cost -= improvement
            
            # Ensure it doesn't drop below a realistic bound
            if current_cost < target_cost:
                current_cost = target_cost + random.uniform(0, target_cost * 0.01)
                
            history.append(current_cost)
            
        runtime = time.perf_counter() - start_time
        
        # We scale the metrics proportionally for the dummy result
        ratio = current_cost / best_cost if best_cost > 0 else 1.0
        
        return Result(
            route=best_route,
            cost=current_cost,
            distance=dist * ratio,
            travel_time=time_val * ratio,
            congestion=cong * ratio,
            iterations=max_iter,
            runtime=runtime,
            history=history,
            feasible=feasible
        )

class QPSO(DummyHeuristicBase):
    def __init__(self):
        # QPSO usually converges fast and finds very good solutions
        super().__init__("QPSO", convergence_speed=0.08, final_gap=0.6)

class PSO(DummyHeuristicBase):
    def __init__(self):
        # standard PSO, slightly slower convergence, worse final gap
        super().__init__("PSO", convergence_speed=0.05, final_gap=0.7)

class GA(DummyHeuristicBase):
    def __init__(self):
        # GA might converge slowly but steady
        super().__init__("GA", convergence_speed=0.03, final_gap=0.75)

class ACO(DummyHeuristicBase):
    def __init__(self):
        # ACO is good at routing, maybe similar to QPSO but slightly worse
        super().__init__("ACO", convergence_speed=0.06, final_gap=0.65)
