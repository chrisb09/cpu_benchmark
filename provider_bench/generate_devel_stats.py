#!/usr/bin/env python3
import pandas as pd
import numpy as np

def calculate_stats(values, prefix):
    n = len(values)
    if n == 0:
        return {}
    
    mean = np.mean(values)
    median = np.median(values)
    min_val = np.min(values)
    max_val = np.max(values)
    
    q25 = np.percentile(values, 25)
    q75 = np.percentile(values, 75)
    iqr = q75 - q25
    
    whis_lower = q25 - 1.5 * iqr
    whis_upper = q75 + 1.5 * iqr
    
    std = np.std(values, ddof=1) if n > 1 else 0.0
    ci_margin = 1.96 * (std / np.sqrt(n)) if n > 0 else 0.0
    ci_lower = mean - ci_margin
    ci_upper = mean + ci_margin
    
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_median": median,
        f"{prefix}_min": min_val,
        f"{prefix}_max": max_val,
        f"{prefix}_p25": q25,
        f"{prefix}_p75": q75,
        f"{prefix}_std_dev": std,
        f"{prefix}_ci_lower": ci_lower,
        f"{prefix}_ci_upper": ci_upper,
        f"{prefix}_whisker_lower": whis_lower,
        f"{prefix}_whisker_upper": whis_upper
    }

def main():
    csv_path = "results_devel/provider_sweep.csv"
    df = pd.read_csv(csv_path)
    
    # Filter for successful runs
    df = df[df['status'] == 'SUCCESS'].copy()
    
    # Convert warm_time_s and job_mem_mb to float
    df['warm_time_s'] = pd.to_numeric(df['warm_time_s'], errors='coerce')
    df['job_mem_mb'] = pd.to_numeric(df['job_mem_mb'], errors='coerce')
    
    stats_list = []
    
    # Group by model and provider
    grouped = df.groupby(['model', 'provider'])
    
    for (model, provider), group in grouped:
        times = group['warm_time_s'].dropna().values
        mems = group['job_mem_mb'].dropna().values
        
        n_time = len(times)
        n_mem = len(mems)
        
        if n_time == 0 or n_mem == 0:
            continue
            
        row_dict = {
            "model": model,
            "provider": provider,
            "runs": n_time
        }
        
        row_dict.update(calculate_stats(times, "time"))
        row_dict.update(calculate_stats(mems, "mem"))
        
        stats_list.append(row_dict)
        
    stats_df = pd.DataFrame(stats_list)
    output_path = "results_devel/provider_stats.csv"
    stats_df.to_csv(output_path, index=False)
    print(f"Statistics generated successfully: {output_path}")
    print(stats_df.to_string(index=False))

if __name__ == "__main__":
    main()
