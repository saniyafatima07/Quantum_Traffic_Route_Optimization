import numpy as np
from benchmark.datasets.problem import Problem

def generate_dataset(size: str, seed: int = 42) -> Problem:
    """
    Generate synthetic VRP datasets of varying sizes.
    """
    np.random.seed(seed)
    
    configs = {
        "small": {"nodes": 20, "vehicles": 5, "deliveries": 15, "capacity": 50},
        "medium": {"nodes": 100, "vehicles": 10, "deliveries": 50, "capacity": 100},
        "large": {"nodes": 500, "vehicles": 25, "deliveries": 250, "capacity": 200},
        "very_large": {"nodes": 1000, "vehicles": 50, "deliveries": 500, "capacity": 300},
    }
    
    if size not in configs:
        raise ValueError(f"Unknown size: {size}. Choose from {list(configs.keys())}")
        
    config = configs[size]
    n = config["nodes"]
    
    # Generate random node coordinates in a 100x100 grid
    coords = np.random.uniform(0, 100, size=(n, 2))
    
    # Calculate euclidean distance matrix
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_matrix[i, j] = np.linalg.norm(coords[i] - coords[j])
                
    # Time matrix: base time is roughly proportional to distance (speed = 1 unit/time)
    time_matrix = dist_matrix * (1.0 + np.random.uniform(-0.1, 0.1, size=(n, n)))
    
    # Generate random demands (node 0 is depot with 0 demand)
    demands = np.random.randint(1, 10, size=n)
    demands[0] = 0
    
    # Ensure some nodes have 0 demand if deliveries < nodes - 1
    if config["deliveries"] < n - 1:
        zero_demand_indices = np.random.choice(range(1, n), size=(n - 1 - config["deliveries"]), replace=False)
        demands[zero_demand_indices] = 0
        
    return Problem(
        name=size,
        num_nodes=n,
        num_vehicles=config["vehicles"],
        num_deliveries=config["deliveries"],
        vehicle_capacity=config["capacity"],
        distance_matrix=dist_matrix,
        time_matrix=time_matrix,
        demands=demands
    )
