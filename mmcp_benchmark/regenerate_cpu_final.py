#!/usr/bin/env python3
"""Strict local m-aia final CPU selection, numeric Cube diagnostics, and figures."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import re
import sys
import tarfile
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

from parse_ml_step_timing import LINE, parse_solver_log

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "results/cpu_final_20261013/maia"
VARIANTS = ("AIX", "blocking", "readiness", "SS_DEFAULT", "SS_TPQ1_I96_B96",
            "SS_TPQ12_I8_B96", "SS_TPQ96_I1_B96")
LABELS = ("AIX", "PhyDLL\nblocking", "PhyDLL\nreadiness", "SS\ndefault",
          "SS\nTPQ1/I96", "SS\nTPQ12/I8", "SS\nTPQ96/I1")
COLORS = dict(zip(VARIANTS, ("#2166ac", "#1b7837", "#5aae61", "#762a83",
                            "#9970ab", "#c2a5cf", "#d6604d")))
WORKLOADS = ("transformer", "mlp_ov0", "mlp_ov4")
TITLES = ("Transformer (ov0)", "MLP (ov0)", "MLP (ov4)")
# Columns follow VARIANTS. Explicit final IDs, never newest-file selection.
# Original 49403xx matrix logs plus CPU_COMPLETION_20261012.md replacements.
JOBS = {
    ("transformer", "timing"): (4952696, 4940320, 4940323, 4966829, 4966830, 4966831, 4966832),
    ("transformer", "profile"): (4940321, 4940324, 4940327, 4957653, 4957655, 4940337, 4940339),
    ("transformer", "memory"): (4940322, 4940325, 4940328, 4957652, 4940335, 4940338, 4940340),
    ("mlp_ov0", "timing"): (4952712, 4940326, 4940329, 4952716, 4952718, 4952719, 4952721),
    ("mlp_ov0", "profile"): (4940341, 4940343, 4958563, 4940347, 4940349, 4940351, 4940353),
    ("mlp_ov0", "memory"): (4940342, 4940344, 4940346, 4940348, 4958564, 4940352, 4940354),
    ("mlp_ov4", "timing"): (4952723, 4952726, 4952727, 4952728, 4952729, 4952730, 4952731),
    ("mlp_ov4", "profile"): (4952697, 4952699, 4952704, 4952706, 4952710, 4952713, 4952717),
    ("mlp_ov4", "memory"): (4952698, 4952701, 4952703, 4958565, 4958566, 4952714, 4952715),
}
CMI_PARSER = HERE.parents[1] / "CPP-ML-Interface/scripts/analyze_cmi_scorep_profiles.py"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def metadata(path):
    result = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            if re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
                require(key not in result, f"duplicate metadata {key}: {path}")
                result[key] = value
    return result


def validate_cell(meta, workload, mode, variant, samples):
    expected = {
        "MEASUREMENT": mode, "WARMUP_CALLS": "2", "SOLVER_RANKS": "96",
        "RUN_STEPS": "80" if mode == "memory" else "200",
        "EXPECTED_INFERENCE_CALLS": "3" if mode == "memory" else "7",
        "ML_CUBE_OVERLAP": "4" if workload == "mlp_ov4" else "0",
        "PROVIDER": "AIX" if variant == "AIX" else
                    "PHYDLL" if variant in ("blocking", "readiness") else "SMARTSIM",
        "SOLVER_EXIT_STATUS": "0", "EXIT_STATUS": "0",
    }
    if variant in ("blocking", "readiness"):
        expected.update(PHYDLL_MODE=variant, DL_RANKS="96")
    if variant.startswith("SS_"):
        expected["SS_CONFIG"] = variant
    for key, value in expected.items():
        require(meta.get(key) == value, f"{workload}/{mode}/{variant}: {key}={meta.get(key)!r}, expected {value}")
    model = "transformer_inference_scripted_fw2.pt" if workload == "transformer" else "mmcp_test_mlp_m5_cpu.pt"
    require(Path(meta["MODEL"]).name == model, "model mismatch")
    build = "build_cpu_retest_scorep" if mode == "profile" else "build_cpu_retest_plain"
    require(build in Path(meta["MAIA"]).parts, "wrong instrumentation build")
    count = int(expected["EXPECTED_INFERENCE_CALLS"])
    require([r["step"] for r in samples] == [15 + 29 * i for i in range(count)], "call schedule mismatch/duplicates")
    for row in samples:
        require(all(math.isfinite(row[k]) and row[k] > 0 for k in ("ml_ms", "rank_min_ms", "rank_max_ms")), "invalid timing")
        require(row["rank_min_ms"] <= row["rank_max_ms"] and
                math.isclose(row["ml_ms"], row["rank_max_ms"], abs_tol=0.002), "timing is not solver-rank maximum")
    if mode == "memory":
        require(meta.get("MEMORY_VALID") == "true", "invalid memory")
        require(meta.get("MEMORY_SCOPE") == "single-cell-job-cgroup-including-startup-and-page-cache", "memory scope")
        require(int(meta["MEMORY_PEAK_BYTES"]) > 0, "nonpositive memory")
        require(Path(meta["MEM_CGROUP_PEAK_PATH"]).name in ("memory.peak", "memory.max_usage_in_bytes"), "not kernel peak")


def select(jobs=JOBS, logs=HERE / "logs"):
    require(set(jobs) == {(w, m) for w in WORKLOADS for m in ("timing", "profile", "memory")}, "incomplete matrix")
    ids, runs, rows, timings, memory = set(), set(), [], [], []
    for (workload, mode), jobids in jobs.items():
        require(len(jobids) == 7, "missing variants")
        for variant, job in zip(VARIANTS, jobids):
            require(job not in ids, "duplicate job ID")
            ids.add(job)
            log = logs / f"retest_{mode}_{workload}_{variant}_96p_{job}.out"
            text = log.read_text()
            logmeta = metadata(log)
            require("VERIFIED " in text and "Preserved bench artifacts:" in text, f"no success marker: {log}")
            run = Path(logmeta["RUN_DIR"])
            require(run.resolve() not in runs, "duplicate run directory")
            runs.add(run.resolve())
            require(re.search(rf"_{job}_[A-Za-z0-9]+$", run.name), "run/job mismatch")
            meta = metadata(run / "run.info")
            for key, value in logmeta.items():
                require(meta.get(key) == value, f"log/run metadata mismatch: {key}")
            samples = parse_solver_log(run / "solver.log")
            require([int(m.group(1)) for m in LINE.finditer((run / "solver.log").read_text())] ==
                    [r["step"] for r in samples], "out-of-order call schedule")
            validate_cell(meta, workload, mode, variant, samples)
            if mode == "memory":
                require(Path(meta["MEM_CGROUP_PEAK_PATH"]).parent.name == f"job_{job}",
                        "not the selected whole-job cgroup (task/node counters rejected)")
            row = dict(workload=workload, mode=mode, variant=variant, job=job, run=str(run),
                       log=str(log), stderr=str(log.with_suffix(".err")), model=meta["MODEL"],
                       overlap=meta["ML_CUBE_OVERLAP"], provider=meta["PROVIDER"],
                       solver_ranks=96, dl_ranks=int(meta.get("DL_RANKS", 0)),
                       warmups=2, calls=len(samples), build=meta["MAIA"],
                       success="local EXIT_STATUS=0 + SOLVER_EXIT_STATUS=0 + VERIFIED",
                       cubex=str(run / "scorep/profile.cubex") if mode == "profile" else "")
            rows.append(row)
            if mode == "timing":
                timings.extend(dict(workload=workload, variant=variant, job=job, sample=i,
                                    steady=i >= 2, **r) for i, r in enumerate(samples))
            if mode == "memory":
                memory.append(dict(workload=workload, variant=variant, job=job,
                                   peak_bytes=int(meta["MEMORY_PEAK_BYTES"]),
                                   peak_GiB=int(meta["MEMORY_PEAK_BYTES"]) / 2**30,
                                   scope=meta["MEMORY_SCOPE"], kernel_counter=meta["MEM_CGROUP_PEAK_PATH"]))
    require(len(rows) == 63, "not 63 cells")
    return rows, timings, memory


def normalize(inclusive, exclusive, visits, rank_count):
    require(rank_count > 0 and visits >= 0, "invalid normalization denominator")
    require(all(math.isfinite(v) and v >= 0 for v in (inclusive, exclusive, visits)), "invalid numeric Cube metric")
    require(exclusive <= inclusive + max(1e-7, inclusive * 1e-4), "exclusive exceeds inclusive")
    return dict(inclusive_rankavg_s=inclusive / rank_count, exclusive_rankavg_s=exclusive / rank_count,
                visits_rankavg=visits / rank_count,
                inclusive_ms_per_visit=1000 * inclusive / visits if visits else "",
                exclusive_ms_per_visit=1000 * exclusive / visits if visits else "")


def extract_profiles(manifest):
    spec = importlib.util.spec_from_file_location("maia_cmi_parser", CMI_PARSER)
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    raw, regions, bins, inventory = [], [], [], []
    for row in manifest:
        if row["mode"] != "profile":
            continue
        cube = Path(row["cubex"])
        with tarfile.open(cube) as archive:
            xml = ET.parse(archive.extractfile("anchor.xml")).getroot()
        groups = xml.findall(".//locationgroup")
        processes = [g for g in groups if g.findtext("type") == "process"]
        count = row["solver_ranks"] + row["dl_ranks"]
        require(sorted(int(g.findtext("rank")) for g in processes) == list(range(count)), "Cube MPI topology mismatch")
        # CMI parser returns trees sorted by location ID, not MPI rank. Map explicitly.
        location_rank = {}
        for g in processes:
            for loc in g.findall("location"):
                require(loc.findtext("type") == "thread", "unexpected location type")
                location_rank[int(loc.attrib["Id"])] = int(g.findtext("rank"))
        require(len(location_rank) == count and len(set(location_rank.values())) == count, "need one thread per MPI rank")
        trees = parser.parse_cubex_file(cube)
        require(len(trees) == count and all(t for t in trees), "empty/missing Cube metrics")
        base = {k: row[k] for k in ("workload", "variant", "job", "cubex")}
        names = sorted({n.name for n in trees[0].values()})
        inventory.append(dict(**base, regions=names, steady_scope_present="solver_step_ml_steady" in names))
        require("m-AIA-Solver::run" in names, "missing MAIA root")
        if row["dl_ranks"]:
            require({"dl_recv", "dl_send", "dl_torch_forward"} <= set(names), "missing DL regions")
        ranktrees = dict(zip((location_rank[i] for i in sorted(location_rank)), trees))
        for scope, ranks, root_name in (("solver_whole_run", range(96), "m-AIA-Solver::run"),
                                        ("dl_whole_client", range(96, count), "phydll_dl_client")):
            if not ranks:
                continue
            aggregates = {}
            totals = {k: 0.0 for k in ("extraction", "reconstruction", "MPI", "other_exclusive")}
            root_total = 0.0
            for rank in ranks:
                tree = ranktrees[rank]
                roots = [n for n in tree.values() if n.name == root_name and n.visits > 0]
                require(len(roots) == 1, f"{row['job']}: root absent on rank {rank}")
                root = roots[0]
                require(root.incl_time > 0, "zero-time Cube root")
                root_total += root.incl_time
                included = []
                for node in tree.values():
                    normalize(node.incl_time, node.excl_time, node.visits, 1)
                    if node.children_ids:
                        childsum = sum(tree[c].incl_time for c in node.children_ids)
                        require(math.isclose(node.incl_time, node.excl_time + childsum, rel_tol=2e-4, abs_tol=2e-5),
                                f"Cube tree algebra mismatch job={row['job']} rank={rank} node={node.name}")
                    current = node
                    while current.id != root.id and current.parent_id is not None:
                        current = tree[current.parent_id]
                    if current.id != root.id:
                        continue
                    included.append(node)
                    raw.append(dict(**base, scope=scope, rank=rank, cnode=node.id, parent=node.parent_id,
                                    region=node.name, inclusive_s=node.incl_time,
                                    exclusive_s=node.excl_time, visits=node.visits))
                    values = aggregates.setdefault(node.id, [node, 0.0, 0.0, 0])
                    values[1] += node.incl_time
                    values[2] += node.excl_time
                    values[3] += node.visits
                    key = ("extraction" if node.name == "flowex_extract_cubes" else
                           "reconstruction" if node.name == "flowex_reconstruct_output" else
                           "MPI" if node.name.startswith("MPI_") else "other_exclusive")
                    totals[key] += node.excl_time
                require(math.isclose(sum(n.excl_time for n in included), root.incl_time, rel_tol=2e-4, abs_tol=2e-5), "exclusive conservation failed")
            for cid, (node, incl, excl, visits) in aggregates.items():
                regions.append(dict(**base, scope=scope, rank_count=len(ranks), cnode=cid, region=node.name,
                                    **normalize(incl, excl, visits, len(ranks))))
            for key, total in totals.items():
                bins.append(dict(**base, scope=scope, bin=key, rank_count=len(ranks),
                                 exclusive_rankavg_s=total / len(ranks),
                                 root_inclusive_rankavg_s=root_total / len(ranks)))
        print(f"PROFILE {row['job']}: {count} ranks, numeric metrics and exclusive conservation validated", flush=True)
    return raw, regions, bins, inventory


def write_csv(path, rows):
    require(bool(rows), f"empty table {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render(timings, memory, bins, regions, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                         "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "axes.axisbelow": True, "axes.grid": False})
    images = []

    def save(fig, name):
        fig.tight_layout(rect=(0, 0.03, 1, 0.92))
        for suffix in ("png", "pdf"):
            path = output / f"{name}.{suffix}"
            fig.savefig(path, dpi=180, facecolor="white")
            require(path.stat().st_size > 1000, "empty figure")
        with Image.open(output / f"{name}.png") as image:
            pixels = np.asarray(image.convert("RGB"))
            require(pixels.std() > 5 and image.width >= 1200 and image.height >= 600, "blank/small figure")
            images.append(dict(file=f"{name}.png", width=image.width, height=image.height,
                               pixel_std=float(pixels.std()), pdf_bytes=(output / f"{name}.pdf").stat().st_size))
        plt.close(fig)

    for kind in ("latency", "memory", "profile_diagnostics"):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=kind != "profile_diagnostics")
        for ax, workload, title in zip(axes, WORKLOADS, TITLES):
            ax.set_title(title)
            ax.set_xticks(range(1, 8), LABELS, fontsize=8)
            ax.grid(axis="y", alpha=0.2)
            if kind == "latency":
                data = [[r["ml_ms"] for r in timings if r["workload"] == workload and r["variant"] == v and r["steady"]] for v in VARIANTS]
                boxes = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False,
                                   medianprops={"color": "black"})
                for i, (v, vals, box) in enumerate(zip(VARIANTS, data, boxes["boxes"]), 1):
                    box.set_facecolor(COLORS[v])
                    box.set_alpha(0.65)
                    ax.scatter([i + (j - 2) * 0.055 for j in range(5)], vals, c=COLORS[v], s=18, edgecolors="black", linewidths=0.3, zorder=3)
                ax.set_yscale("log")
                ax.set_xticks(range(1, 8), LABELS, fontsize=8)
            elif kind == "memory":
                vals = [next(r["peak_GiB"] for r in memory if r["workload"] == workload and r["variant"] == v) for v in VARIANTS]
                ax.bar(range(1, 8), vals, color=[COLORS[v] for v in VARIANTS], width=0.65)
            else:
                bottom = np.zeros(7)
                for key, color in (("extraction", "#4393c3"), ("reconstruction", "#74add1"), ("MPI", "#d6604d"), ("other_exclusive", "#bdbdbd")):
                    vals = [next(r["exclusive_rankavg_s"] for r in bins if r["workload"] == workload and r["variant"] == v and r["scope"] == "solver_whole_run" and r["bin"] == key) for v in VARIANTS]
                    ax.bar(range(1, 8), vals, bottom=bottom, color=color, label=key, width=0.65)
                    bottom += vals
                ax.legend(fontsize=7)
        titles = {"latency": "Steady ML-call latency: 5 calls after 2 warmups; one independent run per cell",
                  "memory": "Kernel whole-job memory peak: one independent job per cell",
                  "profile_diagnostics": "RAW PROFILE DIAGNOSTICS: solver whole-run exclusive bins, not inference phases"}
        axes[0].set_ylabel({"latency": "Solver-rank maximum latency (ms, log scale)", "memory": "Kernel job-cgroup peak (GiB)",
                            "profile_diagnostics": "Solver-rank average time (s)"}[kind])
        fig.suptitle(titles[kind], fontsize=13)
        fig.text(0.5, 0.01, "Profiles include startup and all calls including warmups; no Redis internals." if kind == "profile_diagnostics" else
                 "Boxes summarize within-run calls, not replicated-run confidence intervals." if kind == "latency" else
                 "Includes startup, all processes, runtime and charged page cache; not inference-only RSS.", ha="center", fontsize=9)
        save(fig, kind)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, workload, title in zip(axes, WORKLOADS, TITLES):
        for offset, region, color in ((-0.18, "flowex_extract_cubes", "#4393c3"), (0.18, "flowex_reconstruct_output", "#74add1")):
            vals = [sum(r["exclusive_rankavg_s"] for r in regions if r["workload"] == workload and r["variant"] == v and r["region"] == region and r["scope"] == "solver_whole_run") /
                    sum(r["visits_rankavg"] for r in regions if r["workload"] == workload and r["variant"] == v and r["region"] == region and r["scope"] == "solver_whole_run") * 1000 for v in VARIANTS]
            ax.bar(np.arange(1, 8) + offset, vals, width=0.35, color=color, label=region)
        ax.set_title(title)
        ax.set_xticks(range(1, 8), LABELS, fontsize=8)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Exclusive time / actual region visits (ms)")
    fig.suptitle("RAW REGION DIAGNOSTICS: all-call extraction/reconstruction (including warmups)")
    fig.text(0.5, 0.01, "Region visits differ from ML calls. Not a complete ML-phase decomposition or solver-maximum latency.", ha="center", fontsize=9)
    save(fig, "region_visit_diagnostics")
    return images


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--output", type=Path, default=OUTPUT)
    output = args.parse_args().output.resolve()
    require(output == OUTPUT.resolve(), "outputs are restricted to ignored final maia directory")
    manifest, timings, memory = select()
    output.mkdir(parents=True, exist_ok=True)
    captured_warnings = []
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        raw, regions, bins, inventory = extract_profiles(manifest)
        images = render(timings, memory, bins, regions, output)
        captured_warnings = sorted({str(w.message) for w in recorded})
    for name, rows in (("source_manifest", manifest), ("latency_calls", timings), ("memory", memory),
                       ("profile_rank_metrics", raw), ("profile_regions", regions), ("profile_exclusive_bins", bins)):
        write_csv(output / f"{name}.csv", rows)
    sources = {HERE / "CPU_COMPLETION_20261012.md", Path(__file__), CMI_PARSER,
               HERE / "parse_ml_step_timing.py"}
    for row in manifest:
        sources.update((Path(row["log"]), Path(row["stderr"]), Path(row["run"]) / "run.info", Path(row["run"]) / "solver.log"))
        if row["cubex"]:
            sources.add(Path(row["cubex"]))
    hashes = [dict(path=str(p), bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(sources)]
    blockers = ["No reliable MAIA ML-call parent scope or separate steady/warmup scopes: final inference-phase decomposition blocked.",
                "Numerical parity against pre-patch/GPU outputs remains unverified."]
    report = dict(cells=len(manifest), timing_cells=21, steady_calls=105, warmup_calls=42,
                  memory_cells=21, profile_cells=21, profile_rank_rows=len(raw), profile_inventory=inventory,
                  figures=images, warnings=captured_warnings, blockers=blockers,
                  python=sys.executable, command=f"{sys.executable} mmcp_benchmark/regenerate_cpu_final.py",
                  colors=COLORS, source_hashes=hashes)
    (output / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "README.md").write_text(
        "# m-aia Final CPU Figures\n\n"
        f"Regenerate from workspace root: `{report['command']}`. No benchmark reruns.\n\n"
        "63 explicitly selected successful local cells: 3 workloads x 7 variants x timing/profile/memory. "
        "Source manifest records exact jobs, builds, logs and Cube files; validation_report.json hashes all consumed sources. "
        "Selection rejects duplicates, mode/model/overlap mismatches, failed exits, missing success markers, wrong builds, "
        "invalid kernel peaks and incorrect call schedules. Excludes overlap 6, devel and superseded failures. "
        "Success is certified by preserved local runtime exits and verification, not a fresh Slurm accounting query.\n\n"
        "latency.png/pdf: 21 boxes, each five sequential steady calls (steps 73,102,131,160,189), "
        "discarding two warmups (15,44). One independent run per cell, no replicated-run CI. "
        "ml_ms is the solver-rank maximum; all 147 calls are in latency_calls.csv.\n\n"
        "memory.png/pdf: 21 kernel whole-job cgroup high-water marks / 2^30 GiB, from separate 80-step jobs "
        "with three calls. Includes startup, ranks, controller/server, runtime and charged page cache, not RSS or inference-only memory.\n\n"
        "profile_diagnostics.png/pdf: 21 raw solver-whole-run stacks. Each cnode's exclusive time is assigned "
        "once to extraction, reconstruction, MPI or other. Totals conserve the MAIA root inclusive time. "
        "These include startup/non-ML solver work and all calls including warmups, NOT final inference phases. "
        "Rank metrics are actual Cube time (seconds) and visits; one thread per MPI rank, explicit XML location-to-rank mapping. "
        "Solver and DL ranks are averaged separately (96 each), never merged or treated as solver maxima. "
        "Inclusive and exclusive raw per-rank and rank-average tables retain cnode IDs; inclusive ancestors are never stacked.\n\n"
        "region_visit_diagnostics.png/pdf: actual exclusive time divided by actual visits of each region across "
        "the full solver communicator. Extraction visits are not ML-call counts. No fixed five/seven-call denominator "
        "and no claim of steady scope. profile_regions.csv also reports inclusive/exclusive per-visit metrics "
        "for every recorded scoped cnode, with blank ratios for zero visits. DL-forward visit counts may differ from "
        "solver calls; no assumption that DL regions represent all seven forwards.\n\n"
        "CMI parser is reused read-only for raw inclusive/exclusive/visits values; its automatic step/lifecycle normalization "
        "is deliberately ignored. Numeric finiteness, tree algebra, exclusive conservation and MPI topology are checked. "
        "SmartSim solver/provider client regions cannot measure Redis/RedisAI internals.\n\n"
        "## Outstanding\n\n" + "\n".join(f"- {b}" for b in blockers) + "\n\n"
        f"Image validation: {len(images)} PNG/PDF pairs, dimensions and pixel variation checked. "
        f"Warnings: {json.dumps(captured_warnings)}. See validation_report.json for details.\n")
    print(json.dumps({k: report[k] for k in ("cells", "steady_calls", "profile_rank_rows", "figures", "warnings", "blockers")}, indent=2))


if __name__ == "__main__":
    main()
