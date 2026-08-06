#!/usr/bin/env python3
import csv
import statistics
from collections import defaultdict
from pathlib import Path
from scipy import stats


RESULTS = Path(__file__).with_name("diagnosis_sweep.csv")
OUTPUT = Path(__file__).with_name("diagnosis_stats.csv")


def median(values):
    return statistics.median(values) if values else ""


def main():
    groups = defaultdict(list)
    with RESULTS.open(newline="", encoding="utf-8") as results_file:
        for row in csv.DictReader(results_file):
            groups[row["provider"]].append(row)

    fields = [
        "provider",
        "runs",
        "successful_runs",
        "expected_runs",
        "run_count_note",
        "tpq",
        "intra_threads",
        "bind_cores",
        "cold_median_s",
        "warm_median_s",
        "warm_min_s",
        "warm_max_s",
        "warm_mean_s",
        "warm_std_dev_s",
        "warm_p25_s",
        "warm_p75_s",
        "warm_whisker_lower_s",
        "warm_whisker_upper_s",
        "warm_ci95_lower_s",
        "warm_ci95_upper_s",
        "warm_ci3sigma_lower_s",
        "warm_ci3sigma_upper_s",
        "warm_3sigma_lower_s",
        "warm_3sigma_upper_s",
        "job_mem_median_mb",
    ]
    with OUTPUT.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        for provider, rows in sorted(groups.items()):
            successful = [row for row in rows if row["status"] == "SUCCESS"]
            warm = [float(row["warm_time_s"]) for row in successful]
            cold = [float(row["cold_time_s"]) for row in successful]
            memory = [float(row["job_mem_mb"]) for row in successful]
            
            n = len(warm)
            if n > 0:
                mean_v = statistics.mean(warm)
                std_v = statistics.stdev(warm) if n > 1 else 0.0
                
                # Percentiles and Whiskers
                sorted_warm = sorted(warm)
                # Using simple percentile approximation
                p25_idx = int(0.25 * (n - 1))
                p75_idx = int(0.75 * (n - 1))
                p25_v = sorted_warm[p25_idx]
                p75_v = sorted_warm[p75_idx]
                iqr_v = p75_v - p25_v
                whis_low = p25_v - 1.5 * iqr_v
                whis_high = p75_v + 1.5 * iqr_v
                
                # CIs using Student's t distribution
                se_v = std_v / (n ** 0.5) if n > 0 else 0.0
                if n > 1:
                    t95 = stats.t.ppf(0.975, df=n-1)
                    t3sig = stats.t.ppf(1.0 - (1.0 - 0.9973002039367398)/2.0, df=n-1)
                else:
                    t95 = t3sig = 0.0
                    
                ci95_low = mean_v - t95 * se_v
                ci95_high = mean_v + t95 * se_v
                ci3sig_low = mean_v - t3sig * se_v
                ci3sig_high = mean_v + t3sig * se_v
                sample_3sig_low = mean_v - 3.0 * std_v
                sample_3sig_high = mean_v + 3.0 * std_v
            else:
                mean_v = std_v = p25_v = p75_v = whis_low = whis_high = ""
                ci95_low = ci95_high = ci3sig_low = ci3sig_high = sample_3sig_low = sample_3sig_high = ""

            writer.writerow(
                {
                    "provider": provider,
                    "runs": len(rows),
                    "successful_runs": len(successful),
                    "expected_runs": 5,
                    "run_count_note": "extra run present" if len(rows) != 5 else "",
                    "tpq": rows[0]["tpq"],
                    "intra_threads": rows[0]["intra_threads"],
                    "bind_cores": rows[0]["bind_cores"],
                    "cold_median_s": median(cold),
                    "warm_median_s": median(warm),
                    "warm_min_s": min(warm) if warm else "",
                    "warm_max_s": max(warm) if warm else "",
                    "warm_mean_s": mean_v,
                    "warm_std_dev_s": std_v,
                    "warm_p25_s": p25_v,
                    "warm_p75_s": p75_v,
                    "warm_whisker_lower_s": whis_low,
                    "warm_whisker_upper_s": whis_high,
                    "warm_ci95_lower_s": ci95_low,
                    "warm_ci95_upper_s": ci95_high,
                    "warm_ci3sigma_lower_s": ci3sig_low,
                    "warm_ci3sigma_upper_s": ci3sig_high,
                    "warm_3sigma_lower_s": sample_3sig_low,
                    "warm_3sigma_upper_s": sample_3sig_high,
                    "job_mem_median_mb": median(memory),
                }
            )


if __name__ == "__main__":
    main()
