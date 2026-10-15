#!/usr/bin/env python3
"""Read-only comparison of call 2 against calls 3-7 in final plain runs."""
from __future__ import annotations

import math
from statistics import mean

from regenerate_cpu_final import VARIANTS, WORKLOADS, select


def compare(values):
    if len(values) != 7 or not all(math.isfinite(v) and v > 0 for v in values):
        raise ValueError("expected seven positive finite call durations")
    later = values[2:]
    reference = mean(later)
    return {
        "call2_s": values[1] / 1000,
        "later_mean_s": reference / 1000,
        "later_min_s": min(later) / 1000,
        "later_max_s": max(later) / 1000,
        "call2_delta_pct": 100 * (values[1] / reference - 1),
        "mean_change_pct": 100 * (mean(values[1:]) / reference - 1),
        "call2_above_later_max": values[1] > max(later),
    }


def main():
    # Use the explicit final job manifest and validate the original raw logs;
    # never infer warmup status from the legacy parser's `warm` field.
    manifest, timings, _ = select()
    runs = {(r["workload"], r["variant"]): r for r in manifest if r["mode"] == "timing"}
    assert len(timings) == 147 and len(runs) == 21
    for workload in WORKLOADS:
        print(f"\n### {workload}\n")
        print("| Configuration | Job | Call 2 (s) | Mean 3-7 (s) | Range 3-7 (s) | Call 2 vs mean | Mean change with 2-7 |")
        print("|---|---:|---:|---:|---:|---:|---:|")
        for variant in VARIANTS:
            samples = [r for r in timings if (r["workload"], r["variant"]) == (workload, variant)]
            assert [r["sample"] for r in samples] == list(range(7))
            result = compare([r["ml_ms"] for r in samples])
            print(f"| {variant} | {runs[workload, variant]['job']} | {result['call2_s']:.3f} "
                  f"| {result['later_mean_s']:.3f} "
                  f"| {result['later_min_s']:.3f}-{result['later_max_s']:.3f} "
                  f"| {result['call2_delta_pct']:+.2f}% | {result['mean_change_pct']:+.2f}% |")
            normalized = [r["ml_ms"] / (result["later_mean_s"] * 1000) for r in samples]
            print("<!-- Calls 1-7 / mean(3-7): " + ", ".join(f"{v:.3f}" for v in normalized) + " -->")
    print("\nValidated 21 plain runs / 147 calls from the explicit final job selection.")
    print("Descriptive within-run comparisons only; no independent-run CI or stability test.")
    print("No raw data, metadata, analysis policy, or existing figures modified.")


if __name__ == "__main__":
    main()
