#!/usr/bin/env python3
"""Batch steady extraction for the two-warmup m-aia profiling campaign.

For each replacement profiling job this reads the preserved run directory from
its retest log, requires the engine's VERIFIED marker, and writes combined
``profile_regions.csv`` and ``profile_rank_metrics.csv`` in the new steady
scopes. Only jobs whose logs verify are included; the campaign is incomplete if
any requested cell is missing.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
LOGS = HERE / "logs"
# (workload, variant, dl_ranks) selected for the two-warmup replacement.
JOBS = {
    5007387: ("transformer", "AIX", 0),
    5007388: ("transformer", "blocking", 96),
    5007390: ("transformer", "readiness", 96),
    5007391: ("transformer", "SS_DEFAULT", 0),
    5007392: ("transformer", "SS_TPQ1_I96_B96", 0),
    5007393: ("transformer", "SS_TPQ12_I8_B96", 0),
    5007394: ("transformer", "SS_TPQ96_I1_B96", 0),
    5007395: ("mlp_ov4", "AIX", 0),
    5007396: ("mlp_ov4", "blocking", 96),
    5007397: ("mlp_ov4", "readiness", 96),
    5007398: ("mlp_ov4", "SS_DEFAULT", 0),
    5007399: ("mlp_ov4", "SS_TPQ1_I96_B96", 0),
    5007400: ("mlp_ov4", "SS_TPQ12_I8_B96", 0),
    5007401: ("mlp_ov4", "SS_TPQ96_I1_B96", 0),
}
SOLVER_RANKS = 96


def load_extractor():
    spec = importlib.util.spec_from_file_location("steady_extract", HERE / "extract_steady_profiles.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_dir(log):
    match = re.search(r"^RUN_DIR=(.+)$", log.read_text(), re.MULTILINE)
    if not match:
        raise ValueError(f"no RUN_DIR in {log}")
    return Path(match.group(1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="directory for the two CSVs")
    parser.add_argument("--allow-missing", action="store_true", help="skip cells still running")
    args = parser.parse_args()
    extract = load_extractor()
    verify = extract._verify_module()

    regions, ranks, done = [], [], []
    for job, (workload, variant, dl_ranks) in JOBS.items():
        matches = sorted(LOGS.glob(f"retest_profile_*_{job}.out"))
        if len(matches) != 1:
            raise ValueError(f"expected one log for {job}, found {len(matches)}")
        log = matches[0]
        if "VERIFIED" not in log.read_text():
            if args.allow_missing:
                print(f"SKIP {job} {workload}/{variant}: not verified yet")
                continue
            raise ValueError(f"{job} {workload}/{variant} has no VERIFIED marker")
        directory = run_dir(log)
        r, k, frames = extract.extract_cube(verify, directory / "scorep/profile.cubex",
                                            workload, variant, job, SOLVER_RANKS, dl_ranks)
        regions += r
        ranks += k
        done.append(dict(job=job, workload=workload, variant=variant, dl_ranks=dl_ranks,
                         run=str(directory), frames=frames.get(extract.SOLVER_SCOPE),
                         dl_frames=frames.get(extract.DL_SCOPE, "")))
        print(f"OK {job} {workload}/{variant}: {len(r)} regions frames={frames}")

    args.output.mkdir(parents=True, exist_ok=True)
    extract.write_csv(args.output / "profile_regions.csv", extract.REGION_FIELDS, regions)
    extract.write_csv(args.output / "profile_rank_metrics.csv", extract.RANK_FIELDS, ranks)
    with (args.output / "steady_cells.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(done[0]))
        writer.writeheader()
        writer.writerows(done)
    print(f"Wrote {len(done)} cells, {len(regions)} region rows, {len(ranks)} rank rows to {args.output}")


if __name__ == "__main__":
    main()
