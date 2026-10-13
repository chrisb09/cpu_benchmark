#!/usr/bin/env python3
"""Parse m-aia ``ML_STEP_TIMING`` lines from a benchmark results tree.

The instrumentation (see instrumentation/README.md) prints, on each inference
step from solver rank 0:

    ML_STEP_TIMING step=<globalTimeStep> ml_ms=<max> rank_min_ms=<min> rank_max_ms=<max>

This script walks a results tree for ``solver.log`` files, pairs them with the
``run.info`` metadata written by the runners, drops the cold first inference
(step 15) and reports the warm-inference latency, matching the terrain-solver
campaign's steady-step treatment.

Usage:
    python3 parse_ml_step_timing.py results/characterization_12345 \
        --csv out/timings.csv --summary out/summary.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

LINE = re.compile(
    r"ML_STEP_TIMING\s+step=(\d+)\s+ml_ms=([0-9.]+)\s+"
    r"rank_min_ms=([0-9.]+)\s+rank_max_ms=([0-9.]+)"
)
WARM_FIRST_INFERENCE = 15  # drop the cold first inference, keep step >= 51


def parse_run_info(path: Path) -> dict:
    meta: dict[str, str] = {}
    if not path.is_file():
        return meta
    for line in path.read_text(errors="ignore").splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            meta[key.strip()] = value.strip()
    return meta


def parse_solver_log(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(errors="ignore").splitlines():
        m = LINE.search(line)
        if not m:
            continue
        step = int(m.group(1))
        rows.append(
            dict(
                step=step,
                ml_ms=float(m.group(2)),
                rank_min_ms=float(m.group(3)),
                rank_max_ms=float(m.group(4)),
                warm=step > WARM_FIRST_INFERENCE,
            )
        )
    rows.sort(key=lambda r: r["step"])
    return rows


def collect(root: Path) -> list[dict]:
    records = []
    for solver_log in sorted(root.rglob("solver.log")):
        run_dir = solver_log.parent
        meta = parse_run_info(run_dir / "run.info")
        rows = parse_solver_log(solver_log)
        warmups = int(meta.get("WARMUP_CALLS", "1"))
        for index, row in enumerate(rows):
            row["warm"] = index >= warmups
        if not rows:
            print(f"[warn] no ML_STEP_TIMING lines in {solver_log}", file=sys.stderr)
            continue
        for row in rows:
            records.append(
                dict(
                    run=str(run_dir.relative_to(root)),
                    ranks=meta.get("SOLVER_RANKS", ""),
                    provider=meta.get("PROVIDER", ""),
                    model=Path(meta.get("MODEL", "")).name,
                    steps=meta.get("RUN_STEPS", meta.get("STEPS", "")),
                    wall_s=meta.get("BENCHMARK_SOLVER_WALL_SECONDS", ""),
                    status=meta.get("SOLVER_EXIT_STATUS", ""),
                    **row,
                )
            )
    return records


def write_csv(path: Path, records: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def summarize(records: list[dict]) -> list[dict]:
    groups: dict[tuple, list[float]] = {}
    for row in records:
        if not row["warm"]:
            continue
        key = (row["run"], row["ranks"], row["provider"], row["model"], row["steps"])
        groups.setdefault(key, []).append(row["ml_ms"])
    out = []
    for (run, ranks, provider, model, steps), values in sorted(groups.items()):
        out.append(
            dict(
                run=run,
                ranks=ranks,
                provider=provider,
                model=model,
                steps=steps,
                warm_samples=len(values),
                ml_ms_mean=round(sum(values) / len(values), 3),
                ml_ms_min=round(min(values), 3),
                ml_ms_max=round(max(values), 3),
            )
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="results tree to scan for solver.log")
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    args = parser.parse_args()

    if not args.root.is_dir():
        print(f"not a directory: {args.root}", file=sys.stderr)
        return 2

    records = collect(args.root)
    if not records:
        print("no timing records found", file=sys.stderr)
        return 1

    fields = list(records[0].keys())
    summary = summarize(records)

    if args.csv:
        write_csv(args.csv, records, fields)
        print(f"wrote {args.csv} ({len(records)} rows)")
    if args.summary:
        write_csv(args.summary, summary, list(summary[0].keys()) if summary else [])
        print(f"wrote {args.summary} ({len(summary)} runs)")

    for row in summary:
        print(
            f"{row['run']:>24} ranks={row['ranks']:>3} {row['provider']:>6} "
            f"{row['model']:>36} warm={row['warm_samples']:>2} "
            f"mean={row['ml_ms_mean']:>10.1f} ms  max={row['ml_ms_max']:>10.1f} ms"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
