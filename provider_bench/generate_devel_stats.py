#!/usr/bin/env python3
import pandas as pd
import numpy as np

def main():
    csv_path = "results_devel/provider_sweep.csv"
    df = pd.read_csv(csv_path)
    
    # Filter for successful runs
    df = df[df['status'] == 'SUCCESS'].copy()
    
    # Convert warm_time_s to float
    df['warm_time_s'] = pd.to_numeric(df['warm_time_s'], errors='coerce')
    
    stats_list = []
    
    # Group by model and provider
    grouped = df.groupby(['model', 'provider'])
    
    for (model, provider), group in grouped:
        times = group['warm_time_s'].dropna().values
        n = len(times)
        if n == 0:
            continue
            
        mean = np.mean(times)
        median = np.median(times)
        min_val = np.min(times)
        max_val = np.max(times)
        
        q25 = np.percentile(times, 25)
        q75 = np.percentile(times, 75)
        iqr = q75 - q25
        
        whis_lower = q25 - 1.5 * iqr
        whis_upper = q75 + 1.5 * iqr
        
        # Standard deviation (sample)
        std = np.std(times, ddof=1) if n > 1 else 0.0
        
        # 95% Confidence Interval (Normal approximation matching repository convention)
        ci_margin = 1.96 * (std / np.sqrt(n)) if n > 0 else 0.0
        ci_lower = mean - ci_margin
        ci_upper = mean + ci_margin
        
        stats_list.append({
            "model": model,
            "provider": provider,
            "runs": n,
            "mean": mean,
            "median": median,
            "min": min_val,
            "max": max_val,
            "p25": q25,
            "p75": q75,
            "std_dev": std,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "whisker_lower": whis_lower,
            "whisker_upper": whis_upper
        })
        
    stats_df = pd.DataFrame(stats_list)
    output_path = "results_devel/provider_stats.csv"
    stats_df.to_csv(output_path, index=False)
    print(f"Statistics generated successfully: {output_path}")
    print(stats_df.to_string(index=False))

if __name__ == "__main__":
    main()
