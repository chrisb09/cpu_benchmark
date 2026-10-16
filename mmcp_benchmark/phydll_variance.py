#!/usr/bin/env python3
"""Diagnostic-only PhyDLL profile repeat analysis; writes only CAMPAIGN/analysis.

Run with the Score-P environment used for profiling (cube_dump must be on PATH):
  source /path/to/site/scorep-environment.sh
  python mmcp_benchmark/phydll_variance.py --campaign diagnostics/phydll_variance_20261016 --allow-missing --plots
Matplotlib is needed only with --plots. No jobs are submitted.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import re
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASELINES = (5007388, 5007390, 5007396, 5007397)
CELLS = {(w, v) for w in ("transformer", "mlp_ov4") for v in ("blocking", "readiness")}
REGIONS = {
    "solver_steady": ("cmi_inference_steady", "phydll_prepack", "phydll_send",
                      "phydll_recv", "phydll_unpack", "flowex_extract_cubes",
                      "flowex_reconstruct_output"),
    "dl_steady": ("dl_inference_steady", "dl_torch_forward", "dl_recv",
                  "dl_frame_copy", "dl_send_output"),
}
CORE = {"cmi_inference_steady", "dl_inference_steady", "dl_torch_forward"}
IDENTITY = ("job", "workload", "variant", "rep", "source", "run")
RANK_FIELDS = (*IDENTITY, "scope", "region", "rank", "total_s", "ms_per_call")
PHASE_FIELDS = (*IDENTITY, "scope", "region", "status", "rank_count", "mean_ms",
                "std_ms", "p10_ms", "p50_ms", "p90_ms", "max_ms", "maxrank", "max_over_mean")
CALL_FIELDS = (*IDENTITY, "step", "ml_ms", "rank_max_ms")
RUN_FIELDS = (*IDENTITY, "calls", "mean_ms", "min_ms", "max_ms", "std_ms")
SUMMARY_FIELDS = ("workload", "variant", "scope", "region", "n", "mean_ms", "std_ms",
                  "cv", "ci_half_ms", "ci_status")


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def finite(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"invalid nonnegative finite metric: {value}")
    return value


def percentile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def phase_metrics(rows, frames):
    """Validate selected scoped regions, returning rank means and distributions.

    Multiple cnodes with the same selected name are ambiguous, not additive.
    Optional absent regions get a status row; partial rank coverage is invalid.
    """
    if frames != {"solver_steady": 5, "dl_steady": 5}:
        raise ValueError(f"expected exactly five frames in both scopes: {frames}")
    ranks, phases = [], []
    for scope, names in REGIONS.items():
        expected = set(range(96)) if scope == "solver_steady" else set(range(96, 192))
        for name in names:
            selected = [r for r in rows if r["scope"] == scope and r["region"] == name]
            base = dict(scope=scope, region=name)
            if not selected:
                if name in CORE:
                    raise ValueError(f"missing core region {scope}/{name}")
                phases.append(dict(base, status="missing_optional", rank_count=0))
                continue
            if len(selected) != 96 or {r["rank"] for r in selected} != expected:
                raise ValueError(f"duplicate or incomplete ranks for {scope}/{name}")
            if len({r["cnode"] for r in selected}) != 1:
                raise ValueError(f"ambiguous cnodes for {scope}/{name}")
            values = []
            for row in selected:
                visits = finite(row["visits"])
                if name in ("cmi_inference_steady", "dl_inference_steady") and visits != 5:
                    raise ValueError(f"expected five visits for {scope}/{name}")
                total = finite(row["inclusive_s"])
                finite(row["exclusive_s"])
                value = finite(total * 200)
                values.append(value)
                ranks.append(dict(base, rank=row["rank"], total_s=total, ms_per_call=value))
            mean = statistics.mean(values)
            maximum = max(values)
            maxrank = min(r["rank"] for r, v in zip(selected, values) if v == maximum)
            phases.append(dict(base, status="present", rank_count=96, mean_ms=mean,
                               std_ms=statistics.stdev(values), p10_ms=percentile(values, .1),
                               p50_ms=percentile(values, .5), p90_ms=percentile(values, .9),
                               max_ms=maximum, maxrank=maxrank,
                               max_over_mean=maximum / mean if mean else ""))
    return ranks, phases


def manifest(campaign, campaign_module):
    """Require the four preserved baselines; extras can only be reps 2-5."""
    entries = []
    for job in BASELINES:
        workload, variant, dl_ranks = campaign_module.JOBS[job]
        if dl_ranks != 96:
            raise ValueError(f"invalid baseline rank specification: {job}")
        logs = sorted(campaign_module.LOGS.glob(f"retest_profile_*_{job}.out"))
        if len(logs) != 1:
            raise ValueError(f"expected one baseline log for {job}, found {len(logs)}")
        entries.append(dict(job=str(job), workload=workload, variant=variant, rep=1,
                            source="baseline_profile", log=logs[0]))
    path = campaign / "jobs.csv"
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"job", "workload", "variant", "rep", "log"} <= set(reader.fieldnames or []):
            raise ValueError("jobs.csv requires job,workload,variant,rep,log")
        for row in reader:
            rep = int(row["rep"])
            if (row["workload"], row["variant"]) not in CELLS or rep not in range(2, 6):
                raise ValueError(f"invalid extra cell/rep: {row}")
            if not row["job"].isdigit() or not Path(row["log"]).is_absolute():
                raise ValueError(f"invalid job or nonabsolute log: {row}")
            entries.append(dict(job=str(int(row["job"])), workload=row["workload"],
                                variant=row["variant"], rep=rep,
                                source="extra_profile", log=Path(row["log"])))
    jobs, cells = set(), set()
    for row in entries:
        cell = (row["workload"], row["variant"], row["rep"])
        if row["job"] in jobs or cell in cells:
            raise ValueError(f"duplicate job or cellrep: {row}")
        jobs.add(row["job"])
        cells.add(cell)
    return entries


def verified_run(entry, campaign_module, allow_missing):
    log = entry["log"]
    baseline = entry["rep"] == 1
    if not log.is_file():
        if allow_missing and not baseline:
            return None
        raise ValueError(f"missing log: {log}")
    text = log.read_text()
    if not re.search(r"^VERIFIED(?:\s|$)", text, re.MULTILINE):
        if allow_missing and not baseline:
            return None
        raise ValueError(f"no VERIFIED marker: {log}")
    if len(re.findall(r"^RUN_DIR=.+$", text, re.MULTILINE)) != 1:
        raise ValueError(f"expected one RUN_DIR: {log}")
    run = campaign_module.run_dir(log)
    if not run.is_absolute() or not run.is_dir():
        raise ValueError(f"invalid preserved run: {run}")
    return run.resolve()


def aggregate(phases, runs, ci):
    groups = {}
    for row in phases:
        if row["status"] == "present":
            key = (row["workload"], row["variant"], row["scope"], row["region"])
            groups.setdefault(key, []).append(finite(row["mean_ms"]))
    for row in runs:
        key = (row["workload"], row["variant"], "synchronized_ml", "ML_STEP_TIMING")
        groups.setdefault(key, []).append(finite(row["mean_ms"]))
    result = []
    for (workload, variant, scope, region), values in sorted(groups.items()):
        n = len(values)
        mean = statistics.mean(values)
        std = statistics.stdev(values) if n > 1 else ""
        half = ci.cell_ci(values)[1] if n == 5 else ""
        result.append(dict(workload=workload, variant=variant, scope=scope, region=region,
                           n=n, mean_ms=mean, std_ms=std,
                           cv=std / mean if n > 1 and mean else "", ci_half_ms=half,
                           ci_status="available_95pct_t_df4" if n == 5 else "unavailable_n_not_5"))
    return result


def plots(output, ranks, phases, runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for workload, variant in sorted(CELLS):
        fig, axes = plt.subplots(3, 4, figsize=(16, 10), squeeze=False)
        for ax, (scope, name) in zip(axes.flat, ((s, n) for s, ns in REGIONS.items() for n in ns)):
            for run in runs:
                if (run["workload"], run["variant"]) != (workload, variant):
                    continue
                values = sorted(r["ms_per_call"] for r in ranks if r["job"] == run["job"]
                                and r["scope"] == scope and r["region"] == name)
                if values:
                    ax.step(values, [(i + 1) / len(values) for i in range(len(values))],
                            where="post", label=f"rep {run['rep']} job {run['job']}")
            ax.set(title=name, xlabel="rank mean (ms/call)", ylabel="ECDF")
            if ax.lines:
                ax.legend(fontsize=6)
            else:
                ax.text(.1, .5, "Optional region absent", transform=ax.transAxes)
        fig.suptitle(f"{workload}/{variant}: five-call rank means, not per-call distributions")
        fig.tight_layout()
        fig.savefig(output / f"ranks_{workload}_{variant}.png", dpi=130)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 6))
    for row in phases:
        if row["region"] != "dl_torch_forward" or row["status"] != "present":
            continue
        run = next(r for r in runs if r["job"] == row["job"])
        label = f"{row['workload']}/{row['variant']} r{row['rep']} job {row['job']}"
        ax.scatter(row["mean_ms"], run["mean_ms"], marker="o")
        ax.scatter(row["max_ms"], run["mean_ms"], marker="x")
        ax.annotate(label, (row["max_ms"], run["mean_ms"]), fontsize=6)
    ax.set(xlabel="DL forward: mean (o) / maximum (x) of rank means (ms/call)",
           ylabel="Synchronized ML run mean (ms)",
           title="Profile repeats: maximum of rank means is NOT true per-call critical path")
    fig.tight_layout()
    fig.savefig(output / "latency_vs_forward.png", dpi=150)
    plt.close(fig)


def analyze(campaign, allow_missing=False, make_plots=False):
    campaign = Path(campaign).resolve(strict=True)
    output = campaign / "analysis"
    if output.is_symlink() or any("publication" in part.lower() for part in campaign.parts):
        raise ValueError("analysis must stay in campaign/analysis, outside publication paths")
    campaign_module = load("extract_steady_campaign")
    extract = load("extract_steady_profiles")
    ci = load("maia_latency_ci")
    verify = extract._verify_module()
    entries = manifest(campaign, campaign_module)
    calls, runs, ranks, phases, statuses = [], [], [], [], []
    seen_runs = set()
    for entry in entries:
        run = verified_run(entry, campaign_module, allow_missing)
        status = {k: entry[k] for k in ("job", "workload", "variant", "rep", "source")}
        statuses.append(dict(status, log=str(entry["log"]),
                             status="pending_unverified" if run is None else "analyzed"))
        if run is None:
            continue
        if run in seen_runs:
            raise ValueError(f"duplicate run: {run}")
        seen_runs.add(run)
        identity = dict(status, run=str(run))
        selected = [r for r in ci.calls(run) if r["step"] in ci.STEADY_STEPS]
        mean = finite(ci.run_mean(run))
        if len(selected) != 5:
            raise ValueError(f"expected five steady calls: {run}")
        values = [finite(r["ml_ms"]) for r in selected]
        calls.extend(dict(identity, **r) for r in selected)
        runs.append(dict(identity, calls=5, mean_ms=mean, min_ms=min(values),
                         max_ms=max(values), std_ms=statistics.stdev(values)))
        _, raw, frames = extract.extract_cube(verify, run / "scorep/profile.cubex",
                                              entry["workload"], entry["variant"],
                                              entry["job"], 96, 96)
        rank_rows, phase_rows = phase_metrics(raw, frames)
        ranks.extend(dict(identity, **r) for r in rank_rows)
        phases.extend(dict(identity, **r) for r in phase_rows)
    missing = {(w, v, rep) for w, v in CELLS for rep in range(1, 6)} - {
        (r["workload"], r["variant"], r["rep"]) for r in runs}
    if missing and not allow_missing:
        raise ValueError(f"missing repetitions: {sorted(missing)}")
    summary = aggregate(phases, runs, ci)
    output.mkdir(exist_ok=True)
    for name, fields, rows in (
        ("steady_calls.csv", CALL_FIELDS, calls), ("runs.csv", RUN_FIELDS, runs),
        ("rank_times.csv", RANK_FIELDS, ranks), ("phase_runs.csv", PHASE_FIELDS, phases),
        ("phase_summary.csv", SUMMARY_FIELDS, summary),
        ("run_status.csv", ("job", "workload", "variant", "rep", "source", "log", "status"), statuses)):
        if (output / name).is_symlink():
            raise ValueError(f"refusing symlink output: {output / name}")
        extract.write_csv(output / name, fields, rows)
    report = output / "README.md"
    if report.is_symlink():
        raise ValueError(f"refusing symlink output: {report}")
    report.write_text(
        "# PhyDLL Variance Diagnostics\n\n"
        f"Analyzed {len(runs)} profile runs; missing cell/reps: {sorted(missing)}.\n\n"
        "These are separate profiling repeats, not the plain-timing CI campaign. Rep 1 is the "
        "older preserved two-warmup baseline profile; reps 2-5 are later profiling repeats. "
        "Temporal comparisons are descriptive, not causal evidence for a variant or phase. "
        "Affinity and workload equivalence are not established by this analysis; missing "
        "affinity/workload metadata limits interpretation. No publication files are modified.\n\n"
        "steady_calls.csv contains synchronized ML_STEP_TIMING calls 3-7 from solver.log, "
        "using maia_latency_ci.calls/run_mean. runs.csv reports their mean, min, max and sample "
        "standard deviation. Cube data uses extract_steady_profiles and its dependency-free "
        "cube_dump parser; load/source the site's Score-P environment so cube_dump is on PATH. "
        "Use --campaign DIR [--allow-missing] [--plots]; matplotlib is only needed for plots.\n\n"
        "rank_times.csv contains inclusive scoped totals divided by exactly five calls "
        "(milliseconds). Both scope parents must have five visits; every selected present "
        "region must have 96 ranks. Child visit counts may differ: normalization is by "
        "five inference calls, not by child visits. "
        "Solver and DL scopes are distinct; inclusive parent/child phases overlap and must "
        "not be summed. Missing optional regions are explicit in phase_runs.csv, not zeros. "
        "Core parents and DL forward are mandatory. Rank std is sample std; percentiles use "
        "linear interpolation, tied maxrank selects the lowest rank, and zero-mean ratios "
        "are blank. Aggregate cubes cannot provide per-call phase times or a true per-call "
        "critical path. The maximum of rank means is NOT the mean of per-call maxima.\n\n"
        "phase_summary.csv describes variation of run means (sample std and CV as a ratio), "
        "including baseline rep 1. Only n=5 receives maia_latency_ci.cell_ci's 95% Student-t "
        "interval (df=4); all other CIs are absent. Independence and comparable conditions "
        "are assumptions, not verified by these descriptive intervals. Optional phases may "
        "have smaller n. run_status.csv lists manifest entries skipped as pending; missing "
        "manifest cell/reps are listed above. --allow-missing never excuses invalid baselines "
        "or invalid VERIFIED extra runs. Plots show rank-mean ECDFs and synchronized latency "
        "versus DL-forward mean/maximum of rank means, not per-call critical-path timings.\n")
    if make_plots:
        # Refuse pre-existing plot symlinks just as for the tabular outputs.
        if any(p.is_symlink() for p in output.glob("*.png")):
            raise ValueError("refusing symlink plot output")
        plots(output, ranks, phases, runs)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    print(analyze(args.campaign, args.allow_missing, args.plots))


if __name__ == "__main__":
    main()
