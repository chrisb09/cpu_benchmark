#!/usr/bin/env python3
"""Validate the 96-rank ABBA diagnostic and summarize forward-only clocks."""
import argparse
import csv
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def analyze(root, warmup=3, forwards=9, runs=("1_aix", "2_phydll", "3_phydll", "4_aix"), batch_counts=None):
    batch_counts = batch_counts or {288: 32, 576: 64}
    libraries = []
    for name in ("solver", "dl"):
        text = (root / f"{name}_libraries.txt").read_text()
        assert "not found" not in text and "libscorep" not in text, text
        match = re.search(r"libtorch_cpu\.so\s+=>\s+(\S+)", text)
        assert match, (root, name, "missing CPU libtorch")
        libraries.append(Path(match.group(1)).resolve())
    assert libraries[0] == libraries[1], (root, "CPU libtorch mismatch", libraries)
    rows = []
    summaries = []
    for run in runs:
        directory = root / run / "timings"
        phy = not run.endswith("aix")
        expected_ranks = set(range(192 if phy else 96))
        bindings = list(directory.glob("binding_*.txt"))
        assert len(bindings) == len(expected_ranks), (run, "missing bindings")
        for rank in expected_ranks:
            binding = (directory / f"binding_{rank}.txt").read_text()
            assert binding.rstrip().endswith(f": {rank % 96}"), (run, rank, binding)
        by_rank = defaultdict(list)
        meta = {}
        for path in directory.glob("rank_*_pid_*.txt"):
            rank = int(path.name.split("_")[1])
            for line in path.read_text().splitlines():
                if line.startswith("META "):
                    assert rank not in meta, (run, rank, "duplicate metadata")
                    meta[rank] = line
                elif line.startswith("TIMING "):
                    fields = dict(x.split("=", 1) for x in line.split()[1:])
                    row = dict(run=run, rank=rank, kind=fields["kind"], call=int(fields["call"]))
                    for key in ("wall_s", "thread_s", "process_s"):
                        row[key] = float(fields[key])
                        assert row[key] > 0, (run, rank, row)
                    assert row["process_s"] + .002 >= row["thread_s"], row
                    by_rank[rank].append(row)
        assert set(by_rank) == expected_ranks, (run, "missing timing ranks", set(by_rank))
        for rank, records in by_rank.items():
            assert len(records) == forwards, (run, rank, len(records), forwards)
            expected_kind = "solver_recv" if phy and rank < 96 else "phydll_forward" if phy else "aix_forward"
            assert all(r["kind"] == expected_kind for r in records), (run, rank, records)
            assert sorted(r["call"] for r in records) == list(range(forwards)), (run, rank)
            for row in records:
                row["steady"] = row["call"] >= warmup
                rows.append(row)
            if expected_kind != "solver_recv":
                line = meta[rank]
                assert "dtype=Float device=cpu contiguous=1" in line, line
                assert "model_dtype=Float model_device=cpu" in line, line
                assert f"intra=1 inter={1 if phy else 96} " in line, line
                assert f"affinity={rank % 96}," in line, line
                shape = re.search(r"shape=\[(\d+), 18\]", line)
                assert shape and int(shape.group(1)) in batch_counts, line
                assert "strides=[18, 1]" in line, line
        batches = Counter(int(re.search(r"shape=\[(\d+),", line).group(1)) for line in meta.values())
        assert batches == batch_counts, (run, batches)
        for kind in sorted({r["kind"] for r in rows if r["run"] == run}):
            selected = [r for r in rows if r["run"] == run and r["kind"] == kind and r["steady"]]
            summary = dict(run=run, kind=kind, samples=len(selected))
            for key in ("wall_s", "thread_s", "process_s"):
                summary[f"mean_{key}"] = statistics.mean(r[key] for r in selected)
            summary["thread_wall_ratio"] = summary["mean_thread_s"] / summary["mean_wall_s"]
            summary["process_wall_ratio"] = summary["mean_process_s"] / summary["mean_wall_s"]
            summaries.append(summary)
            print(" ".join(f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}" for k, v in summary.items()))
        counts = Counter(re.sub(r" affinity=\S+", "", line) for line in meta.values())
        for line, count in sorted(counts.items()):
            print(f"{run} metadata_ranks={count} {line}")
    with (root / "timings.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / "summary.json").write_text(json.dumps(dict(warmup_forwards=warmup,
        forwards_per_rank=forwards, summaries=summaries), indent=2) + "\n")
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--forwards", type=int, default=9)
    parser.add_argument("--runs", nargs="+", default=["1_aix", "2_phydll", "3_phydll", "4_aix"])
    args = parser.parse_args()
    analyze(args.root, args.warmup, args.forwards, args.runs)
