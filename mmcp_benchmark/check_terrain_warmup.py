#!/usr/bin/env python3
"""Read-only second-call check across the selected terrain plain repetitions."""
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median
import sys

TERRAIN = Path(__file__).resolve().parents[2] / "mini_app"
sys.path.insert(0, str(TERRAIN))
from terrain_cpu_diagnostics import timings

CONFIGS = ("AIx", "PHYDLL_FALSE", "PHYDLL_TRUE", "SS_DEFAULT",
           "TPQ1_I96", "TPQ12_I8", "TPQ96_I1")


def compare(values):
    # Ten existing selected calls: call 2 at step 4, then calls 3-11.
    if len(values) != 10 or not all(math.isfinite(v) and v > 0 for v in values):
        raise ValueError("expected ten positive finite durations")
    reference = mean(values[1:])
    return dict(call2_delta_pct=100 * (values[0] / reference - 1),
                call2_above_later_max=values[0] > max(values[1:]),
                current_mean_ms=mean(values), later_mean_ms=reference,
                normalized_calls=[v / reference for v in values])


def main():
    manifest = TERRAIN / "campaign_figures/terrain_cpu_complete_20261013/timing_replicates.csv"
    with manifest.open() as stream:
        rows = list(csv.DictReader(stream))
    expected = {(m, c, r) for m in ("watercnn", "giant") for c in CONFIGS for r in range(1, 6)}
    keys = [(r["model"], r["config"], int(r["rep"])) for r in rows]
    if len(keys) != 70 or set(keys) != expected:
        raise ValueError("missing/duplicate/unexpected selected runs")
    results = {}
    for key, row in zip(keys, rows):
        log = Path(row["log"])
        artifacts = json.loads(row["artifacts"])
        digest = hashlib.sha256(log.read_bytes()).hexdigest()
        if not any(a.get("path") == str(log) and a.get("sha256") == digest for a in artifacts):
            raise ValueError(f"source log hash mismatch: {log}")
        samples = timings(log, 22)
        selected = json.loads(row["steady_samples_ms"])
        if {str(s["step"]): s["step_ms"] for s in samples} != selected:
            raise ValueError(f"raw timings differ from selected manifest: {log}")
        values = [s["step_ms"] for s in samples]
        results[key] = compare(values)
    for model in ("watercnn", "giant"):
        print(f"\n### {model}\n")
        print("| Configuration | Median call 2 difference | Replicate range | Above later max | Mean change if excluded |")
        print("|---|---:|---:|---:|---:|")
        for config in CONFIGS:
            runs = [results[model, config, r] for r in range(1, 6)]
            differences = [r["call2_delta_pct"] for r in runs]
            # Campaign latency is the equally weighted mean of independent run means.
            change = 100 * (mean(r["later_mean_ms"] for r in runs) /
                            mean(r["current_mean_ms"] for r in runs) - 1)
            print(f"| {config} | {median(differences):+.2f}% "
                  f"| {min(differences):+.2f}% to {max(differences):+.2f}% "
                  f"| {sum(r['call2_above_later_max'] for r in runs)}/5 | {change:+.2f}% |")
            print("<!-- Replicates 1-5 call 2 differences: " +
                  ", ".join(f"{d:+.3f}%" for d in differences) + " -->")
            print("<!-- Median normalized calls 2-11: " +
                  ", ".join(f"{median(r['normalized_calls'][i] for r in runs):.3f}"
                            for i in range(10)) + " -->")
    print("\nValidated 70 raw log hashes and 700 selected per-call timings against the final manifest.")
    print("Call 2 = step 4; reference = calls 3-11 at steps 6-22.")
    print("Descriptive checks only. No data, figures, or policy changed.")


if __name__ == "__main__":
    main()
