#!/usr/bin/env python3
import csv
import statistics
from collections import defaultdict
from pathlib import Path


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
                    "warm_mean_s": statistics.mean(warm) if warm else "",
                    "warm_std_dev_s": statistics.stdev(warm) if len(warm) > 1 else "",
                    "job_mem_median_mb": median(memory),
                }
            )


if __name__ == "__main__":
    main()
