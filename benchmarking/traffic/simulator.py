import numpy as np

class TrafficSimulator:
    def __init__(self, num_nodes: int, base_time_matrix: np.ndarray, seed: int = 42):
        self.num_nodes = num_nodes
        self.base_time_matrix = base_time_matrix
        self.congestion_matrix = np.zeros((num_nodes, num_nodes))
        self.alpha = 1.0 # Congestion sensitivity
        np.random.seed(seed)
        
    def get_travel_time(self, i: int, j: int) -> float:
        """Calculate dynamic travel time W_ij(t) = D_ij * (1 + alpha * C_ij(t))"""
        base_time = self.base_time_matrix[i, j]
        congestion = self.congestion_matrix[i, j]
        return base_time * (1 + self.alpha * congestion)
        
    def get_all_travel_times(self) -> np.ndarray:
        """Returns the full dynamic time matrix"""
        return self.base_time_matrix * (1 + self.alpha * self.congestion_matrix)
        
    def set_congestion_level(self, level: str):
        """
        Updates the congestion matrix based on a predefined level.
        Levels: 'normal', 'moderate', 'heavy', 'blockage'
        """
        if level == "normal":
            self.congestion_matrix = np.random.uniform(0.0, 0.1, size=(self.num_nodes, self.num_nodes))
        elif level == "moderate":
            self.congestion_matrix = np.random.uniform(0.2, 0.5, size=(self.num_nodes, self.num_nodes))
        elif level == "heavy":
            self.congestion_matrix = np.random.uniform(0.5, 1.5, size=(self.num_nodes, self.num_nodes))
        elif level == "blockage":
            # Add extreme congestion to a few random edges
            self.congestion_matrix = np.random.uniform(0.2, 0.5, size=(self.num_nodes, self.num_nodes))
            num_blocked = max(1, int(self.num_nodes * 0.05))
            for _ in range(num_blocked):
                i, j = np.random.randint(0, self.num_nodes, size=2)
                self.congestion_matrix[i, j] = 10.0 # Huge multiplier
        else:
            raise ValueError(f"Unknown congestion level: {level}")
            
        # Ensure diagonal is zero (no self congestion)
        np.fill_diagonal(self.congestion_matrix, 0)
