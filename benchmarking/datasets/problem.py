import dataclasses
from typing import Dict, List, Tuple
import numpy as np

@dataclasses.dataclass
class Problem:
    name: str
    num_nodes: int
    num_vehicles: int
    num_deliveries: int
    vehicle_capacity: float
    
    # distance_matrix[i][j] = base distance from i to j
    distance_matrix: np.ndarray 
    
    # time_matrix[i][j] = base time from i to j (without traffic)
    time_matrix: np.ndarray 
    
    # Node 0 is usually depot
    demands: np.ndarray # Array of shape (num_nodes,)
    
    def get_base_cost(self, i: int, j: int) -> float:
        return self.distance_matrix[i, j]
