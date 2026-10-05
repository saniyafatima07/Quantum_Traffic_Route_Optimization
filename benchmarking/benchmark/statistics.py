import pandas as pd

def generate_statistics():
    results_file = "benchmark/results/metrics.csv"
    
    try:
        df = pd.read_csv(results_file)
    except FileNotFoundError:
        print("Results file not found. Run benchmark/runner.py first.")
        return
        
    print("====== Benchmarking Statistics ======\n")
    
    # Normal traffic stats
    normal_df = df[df['traffic'] == 'normal']
    
    for dataset in normal_df['dataset'].unique():
        print(f"--- Dataset: {dataset.upper()} ---")
        ds_df = normal_df[normal_df['dataset'] == dataset]
        
        stats = ds_df.groupby('algorithm').agg({
            'cost': ['mean', 'std'],
            'distance': 'mean',
            'travel_time': 'mean',
            'runtime': 'mean'
        }).reset_index()
        
        # Format the output nicely
        stats.columns = ['Algorithm', 'Mean Cost', 'Cost StdDev', 'Mean Dist', 'Mean Time(Obj)', 'Mean Runtime(s)']
        print(stats.to_string(index=False, float_format="%.2f"))
        print("\n")
        
    print("--- Dynamic Traffic (Medium Dataset) ---")
    dyn_df = df[(df['dataset'] == 'medium')]
    
    dyn_stats = dyn_df.groupby(['traffic', 'algorithm']).agg({
        'cost': 'mean',
        'congestion': 'mean'
    }).reset_index()
    
    print(dyn_stats.to_string(index=False, float_format="%.2f"))
    print("\n")
    
if __name__ == "__main__":
    generate_statistics()
