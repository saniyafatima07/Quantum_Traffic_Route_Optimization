import os
import time
import json
import csv
from tqdm import tqdm

from benchmark.datasets.generator import generate_dataset
from benchmark.traffic.simulator import TrafficSimulator
from benchmark.algorithms.qpso import QPSO
from benchmark.algorithms.pso import PSO
from benchmark.algorithms.ga import GA
from benchmark.algorithms.aco import ACO

def run_benchmark():
    datasets = ["small", "medium", "large"]
    # For a quicker test, we can just run 5 seeds
    num_seeds = 5
    max_iterations = 100
    
    algorithms = [QPSO(), PSO(), GA(), ACO()]
    
    os.makedirs("benchmark/results", exist_ok=True)
    results_file = "benchmark/results/metrics.csv"
    history_file = "benchmark/results/history.json"
    
    histories = {}
    
    with open(results_file, 'w', newline='') as csvfile:
        fieldnames = ['dataset', 'nodes', 'algorithm', 'seed', 'traffic', 'cost', 'distance', 'travel_time', 'congestion', 'runtime', 'iterations', 'feasible']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Experiment 1, 2, 3, 4 (Scalability, Quality, Runtime, Convergence)
        print("Running Main Benchmark...")
        for ds_name in datasets:
            problem = generate_dataset(ds_name)
            
            # Initialize Traffic Simulator with normal traffic
            traffic_sim = TrafficSimulator(problem.num_nodes, problem.time_matrix)
            traffic_sim.set_congestion_level("normal")
            
            for algo in algorithms:
                histories.setdefault(ds_name, {}).setdefault(algo.name, {})
                for seed in tqdm(range(1, num_seeds + 1), desc=f"{ds_name} - {algo.name}"):
                    
                    result = algo.solve(problem, traffic_sim=traffic_sim, seed=seed, max_iter=max_iterations)
                    
                    writer.writerow({
                        'dataset': ds_name,
                        'nodes': problem.num_nodes,
                        'algorithm': algo.name,
                        'seed': seed,
                        'traffic': 'normal',
                        'cost': result.cost,
                        'distance': result.distance,
                        'travel_time': result.travel_time,
                        'congestion': result.congestion,
                        'runtime': result.runtime,
                        'iterations': result.iterations,
                        'feasible': result.feasible
                    })
                    
                    # Save history for convergence plots
                    histories[ds_name][algo.name][seed] = result.history
                    
        # Experiment 5: Dynamic Traffic (Testing on medium dataset)
        print("Running Dynamic Traffic Benchmark...")
        problem = generate_dataset("medium")
        traffic_levels = ["normal", "moderate", "heavy", "blockage"]
        
        for level in traffic_levels:
            traffic_sim = TrafficSimulator(problem.num_nodes, problem.time_matrix)
            traffic_sim.set_congestion_level(level)
            
            for algo in algorithms:
                for seed in range(1, 3): # Just a couple runs for traffic robustness
                    result = algo.solve(problem, traffic_sim=traffic_sim, seed=seed, max_iter=max_iterations)
                    
                    writer.writerow({
                        'dataset': 'medium',
                        'nodes': problem.num_nodes,
                        'algorithm': algo.name,
                        'seed': seed,
                        'traffic': level,
                        'cost': result.cost,
                        'distance': result.distance,
                        'travel_time': result.travel_time,
                        'congestion': result.congestion,
                        'runtime': result.runtime,
                        'iterations': result.iterations,
                        'feasible': result.feasible
                    })

    # Save convergence history
    with open(history_file, 'w') as f:
        json.dump(histories, f)
        
    print(f"Benchmark completed. Results saved to {results_file} and {history_file}")

if __name__ == "__main__":
    run_benchmark()
