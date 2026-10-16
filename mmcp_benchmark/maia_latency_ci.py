#!/usr/bin/env python3
"""Run-mean latency CI for m-aia from five independent plain-timing repetitions.

Rep 1 is the existing selected plain-timing run; reps 2-5 come from the CI
campaign logs. Each repetition yields one run mean over the five measured calls
(3-7). A cell's latency is the mean of its five run means with a 95% Student-t
interval (df=4, t=2.776445), matching the terrain method.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results/cpu_final_steady_20261014"
T95 = 2.776445
STEADY_STEPS = (73, 102, 131, 160, 189)
WORKLOADS = ("transformer", "mlp_ov4")
VARIANTS = ("AIX", "blocking", "readiness", "SS_DEFAULT", "SS_TPQ1_I96_B96",
            "SS_TPQ12_I8_B96", "SS_TPQ96_I1_B96")
LINE = re.compile(r"ML_STEP_TIMING step=(\d+) ml_ms=([0-9.]+) rank_min_ms=([0-9.]+) rank_max_ms=([0-9.]+)")


def calls(run):
    rows = []
    for line in (run / "solver.log").read_text().splitlines():
        match = LINE.search(line)
        if match:
            rows.append(dict(step=int(match.group(1)), ml_ms=float(match.group(2)),
                             rank_max_ms=float(match.group(4))))
    rows.sort(key=lambda r: r["step"])
    if [r["step"] for r in rows] != [15 + 29 * i for i in range(7)]:
        raise ValueError(f"unexpected call schedule in {run}: {[r['step'] for r in rows]}")
    for row in rows:
        if not math.isfinite(row["ml_ms"]) or row["ml_ms"] <= 0 or row["ml_ms"] != row["rank_max_ms"]:
            raise ValueError(f"invalid ML-step timing in {run}")
    return rows


def run_mean(run):
    selected = [r for r in calls(run) if r["step"] in STEADY_STEPS]
    if len(selected) != len(STEADY_STEPS):
        raise ValueError(f"missing steady calls in {run}")
    return sum(r["ml_ms"] for r in selected) / len(selected)


def cell_ci(means):
    if len(means) != 5:
        raise ValueError(f"95% t interval needs five independent run means, got {len(means)}")
    mean = sum(means) / 5
    var = sum((m - mean) ** 2 for m in means) / 4
    return mean, T95 * math.sqrt(var / 5)


def _rep1_runs():
    rows = list(csv.DictReader((RESULTS / "maia/source_manifest.csv").open()))
    return {(r["workload"], r["variant"]): Path(r["run"]) for r in rows
            if r["mode"] == "timing" and r["workload"] in WORKLOADS}


def _job_runs(jobs_csv):
    mapping = {}
    for r in csv.DictReader(jobs_csv.open()):
        log = HERE / "logs" / f"ci_{r['workload']}_{r['variant']}_96p_r{r['rep']}_{r['job']}.out"
        if not log.is_file():
            raise ValueError(f"missing CI log {log}")
        match = re.search(r"^RUN_DIR=(.+)$", log.read_text(), re.MULTILINE)
        if not match:
            raise ValueError(f"no RUN_DIR in {log}")
        if "VERIFIED" not in log.read_text():
            raise ValueError(f"{log} not verified")
        mapping.setdefault((r["workload"], r["variant"]), {})[int(r["rep"])] = Path(match.group(1))
    return mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs-csv", type=Path, default=RESULTS / "maia_ci_jobs.csv")
    parser.add_argument("--output", type=Path, default=RESULTS / "maia/maia_latency_summary.csv")
    parser.add_argument("--reps-csv", type=Path, default=RESULTS / "maia/maia_latency_reps.csv")
    args = parser.parse_args()

    rep1 = _rep1_runs()
    extra = _job_runs(args.jobs_csv)
    summary, reps = [], []
    for workload in WORKLOADS:
        for variant in VARIANTS:
            runs = {1: rep1[(workload, variant)]}
            runs.update(extra[(workload, variant)])
            if set(runs) != {1, 2, 3, 4, 5}:
                raise ValueError(f"{workload}/{variant}: reps {sorted(runs)}")
            means = []
            for rep in range(1, 6):
                value = run_mean(reps_run := runs[rep])
                means.append(value)
                reps.append(dict(workload=workload, variant=variant, rep=rep,
                                 run=str(reps_run), run_mean_ms=value))
            mean, half = cell_ci(means)
            summary.append(dict(workload=workload, variant=variant, independent_runs=5,
                                calls_per_run=5, mean_ms=mean, ci_half_ms=half,
                                ci_status="available_95pct_t_df4",
                                samples_ms=json.dumps(means)))
            print(f"{workload:12} {variant:20} mean={mean/1000:9.3f}s  ci=+/-{half/1000:7.3f}s")
    for path, rows in ((args.reps_csv, reps), (args.output, summary)):
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"wrote {len(summary)} cells to {args.output}")


if __name__ == "__main__":
    main()
