from typing import List, Tuple
from benchmark.datasets.problem import Problem
from benchmark.traffic.simulator import TrafficSimulator

def evaluate_route(route: List[int], problem: Problem, traffic_sim: TrafficSimulator = None) -> Tuple[float, float, float]:
    """
    Evaluates a single route (sequence of nodes starting and ending at 0).
    Returns (distance, travel_time, congestion_penalty)
    """
    if len(route) <= 1:
        return 0.0, 0.0, 0.0
        
    dist = 0.0
    time = 0.0
    congestion = 0.0
    
    for i in range(len(route) - 1):
        u, v = route[i], route[i+1]
        dist += problem.distance_matrix[u, v]
        
        if traffic_sim:
            travel_time = traffic_sim.get_travel_time(u, v)
            base_time = problem.time_matrix[u, v]
            time += travel_time
            congestion += (travel_time - base_time)
        else:
            time += problem.time_matrix[u, v]
            
    return dist, time, congestion

def evaluate_solution(routes: List[List[int]], problem: Problem, traffic_sim: TrafficSimulator = None, 
                      w1: float = 1.0, w2: float = 1.0, w3: float = 1.0) -> Tuple[float, float, float, float, bool]:
    """
    Evaluates the full solution (list of routes for each vehicle).
    Returns (total_distance, total_travel_time, congestion_penalty, objective_value, feasible)
    """
    total_dist = 0.0
    total_time = 0.0
    total_congestion = 0.0
    
    feasible = True
    visited = set()
    
    for route in routes:
        # Check if route starts and ends at 0
        if not route or route[0] != 0 or route[-1] != 0:
            feasible = False
            
        # Check capacity
        load = sum(problem.demands[node] for node in route)
        if load > problem.vehicle_capacity:
            feasible = False
            
        for node in route:
            if node != 0:
                visited.add(node)
                
        d, t, c = evaluate_route(route, problem, traffic_sim)
        total_dist += d
        total_time += t
        total_congestion += c
        
    # Check if all demands are met
    # Nodes with demand > 0 must be visited
    for node in range(1, problem.num_nodes):
        if problem.demands[node] > 0 and node not in visited:
            feasible = False
            
    # Calculate objective
    objective = w1 * total_dist + w2 * total_time + w3 * total_congestion
    
    # Penalize infeasible solutions heavily
    if not feasible:
        objective += 100000.0
        
    return total_dist, total_time, total_congestion, objective, feasible
