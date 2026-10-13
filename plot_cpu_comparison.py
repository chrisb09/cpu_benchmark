"""Render CPU comparisons from preserved CSVs only; never launch a solver."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
MAIA = ROOT / "mmcp_benchmark/results/cpu_final_20261013/maia"
TERRAIN = ROOT.parent / "mini_app/campaign_figures/terrain_cpu_complete_20261013"
OUTPUT = ROOT / "figures/cpu_final"
T95 = 2.776445
MAIN = ("AIX", "blocking", "SS_DEFAULT", "SS_TPQ1_I96_B96",
        "SS_TPQ12_I8_B96", "SS_TPQ96_I1_B96")
READY = ("AIX", "blocking", "readiness")
MODELS = ("watercnn", "giant", "transformer", "mlp_ov4")
MODEL_LABELS = {"watercnn": "Terrain: WaterCNN", "giant": "Terrain: Giant MLP",
                "transformer": "MAIA: Transformer OV0", "mlp_ov4": "MAIA: MLP OV4",
                "mlp_ov0": "MAIA: MLP OV0"}
LABELS = {"AIX": "AIx", "blocking": "PhyDLL\nblocking", "readiness": "PhyDLL\nreadiness",
          "SS_DEFAULT": "SmartSim\ndefault", "SS_TPQ1_I96_B96": "SmartSim\nTPQ=1\nINTRA=96",
          "SS_TPQ12_I8_B96": "SmartSim\nTPQ=12\nINTRA=8",
          "SS_TPQ96_I1_B96": "SmartSim\nTPQ=96\nINTRA=1"}
CONFIG = {"AIx": "AIX", "PHYDLL_FALSE": "blocking", "PHYDLL_TRUE": "readiness",
          "TPQ1_I96": "SS_TPQ1_I96_B96", "TPQ12_I8": "SS_TPQ12_I8_B96",
          "TPQ96_I1": "SS_TPQ96_I1_B96"}
COLORS = {"Inference": "#8064A2", "Input receive": "#4C78A8", "Forward": "#F58518",
          "Output send": "#54A24B", "Other": "#BAB0AC", "Put": "#72B7B2",
          "Model request": "#E45756", "Retrieval/unpack": "#B279A2"}
PROVIDER_COLORS = dict(zip((*MAIN, "readiness"),
                          ("#577787", "#C97AA7", "#8064A2", "#72B7B2", "#E2A354", "#B279A2", "#009E73")))
FULL_LEGEND = {"Input receive": "Input receive (DL-side receive + handling)",
               "Forward": "Forward (DL-side model execution)",
               "Output send": "Output send (DL-side output transfer)",
               "Other": "Other (staging / bookkeeping)",
               "Put": "Put (client tensor submission)",
               "Model request": "Model request (client elapsed time)",
               "Retrieval/unpack": "Retrieval/unpack (client retrieval + unpack)",
               "Inference": "Inference (AIx library call)"}
DL_NODES = {"Input receive": ("dl_recv", "dl_frame_copy"),
            "Forward": ("dl_torch_forward",), "Output send": ("dl_send_output",),
            "Other": ("dl_input_allocate", "dl_output_allocate", "dl_input_unpack",
                      "dl_h2d", "dl_d2h", "dl_output_reorder")}
TERRAIN_BINS = {"Input receive": ("dl_recv / dl_frame_copy",),
                "Forward": ("dl_torch_forward",), "Output send": ("dl_send_output",),
                "Other": ("dl_allocate", "dl_input_unpack", "dl_h2d", "dl_d2h",
                          "dl_output_reorder", "Other / overhead")}
SS_BINS = {"Put": ("smartsim_put_tensor",), "Model request": ("smartsim_run_model",),
           "Retrieval/unpack": ("smartsim_unpack_tensor",),
           "Other": ("smartsim_chunk_plan", "smartsim_token_wait", "Other / overhead")}


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def number(value):
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"Invalid nonnegative measurement: {value}")
    return result


def mean_ci(values, independent=True):
    values = np.array([number(v) for v in values])
    if not len(values):
        raise ValueError("Empty measurements")
    mean = float(values.mean())
    if not math.isfinite(mean):
        raise ValueError("Nonfinite mean")
    if not independent or len(values) == 1:
        return mean, None, "unavailable_single_independent_run"
    if len(values) != 5:
        raise ValueError("95% t interval requires five independent run means")
    half = float(T95 * values.std(ddof=1) / math.sqrt(5))
    if not math.isfinite(half):
        raise ValueError("Nonfinite CI")
    return mean, half, "available_95pct_t_df4"


def selected_models(include_ov0=False):
    return MODELS + (("mlp_ov0",) if include_ov0 else ())


def assert_disjoint(nodes, parents):
    selected = set(nodes)
    if len(selected) != len(nodes):
        raise ValueError("Duplicate inclusive node")
    for node in nodes:
        seen = {node}
        parent = parents.get(node)
        while parent is not None and parent != "0":
            if parent in seen or parent in selected:
                raise ValueError("Inclusive ancestor double counting or cycle")
            seen.add(parent)
            parent = parents.get(parent)


def inclusive_phase(rows, names, frames, parents):
    chosen = []
    for name in names:
        matches = [r for r in rows if r["region"] == name]
        if len(matches) != 1:
            raise ValueError(f"Missing or ambiguous region {name}")
        row = matches[0]
        if number(row["visits_rankavg"]) != frames:
            raise ValueError(f"Region {name} does not have {frames} visits/rank")
        chosen.append(row)
    assert_disjoint([r["cnode"] for r in chosen], parents)
    return sum(number(r["inclusive_rankavg_s"]) for r in chosen) * 1000 / frames


def load_data():
    paths = [MAIA / name for name in ("source_manifest.csv", "latency_calls.csv", "memory.csv",
             "profile_regions.csv", "profile_rank_metrics.csv")]
    paths += [TERRAIN / name for name in ("timing_replicates.csv", "memory_replicates.csv",
              "phase_replicates.csv", "profile_manifest.csv")]
    tables = {p.name: read_csv(p) for p in paths}
    if len(tables["source_manifest.csv"]) != 63:
        raise ValueError("MAIA selection is not 63 cells")
    for name in ("timing_replicates.csv", "memory_replicates.csv", "profile_manifest.csv"):
        if len(tables[name]) != 70:
            raise ValueError(f"Terrain {name} is not 70 cells")
    latency, memory, phases, audit, preprocessing = [], [], [], [], []
    for model in selected_models(True):
        terrain = model in MODELS[:2]
        for variant in (*MAIN, "readiness"):
            base = {"model": model, "variant": variant}
            if terrain:
                runs = [r for r in tables["timing_replicates.csv"]
                        if r["model"] == model and CONFIG.get(r["config"], r["config"]) == variant]
                if len(runs) != 5 or len({r["rep"] for r in runs}) != 5:
                    raise ValueError(f"Timing run count {base}")
                for r in runs:
                    if r["status"] != "SUCCESS" or len(json.loads(r["steady_samples_ms"])) != 10:
                        raise ValueError(f"Timing status/schedule {base}")
                    samples = [number(v) for v in json.loads(r["steady_samples_ms"]).values()]
                    if not math.isclose(np.mean(samples), number(r["steady_mean_ms"]), rel_tol=1e-8):
                        raise ValueError("Run mean does not match ten steady samples")
                values = [number(r["steady_mean_ms"]) for r in runs]
                mean, ci, status = mean_ci(values)
                latency.append(dict(base, mean_ms=mean, ci_half_ms=ci, ci_status=status,
                                    independent_runs=5, calls_per_run=10, samples_ms=json.dumps(values),
                                    scope="plain steady ML step"))
                mem = [r for r in tables["memory_replicates.csv"]
                       if r["model"] == model and CONFIG.get(r["config"], r["config"]) == variant]
                if len(mem) != 5 or len({r["rep"] for r in mem}) != 5:
                    raise ValueError(f"Memory run count {base}")
                for kind in ("pss", "rss"):
                    memory.append(dict(base, metric=kind.upper(), value_GiB=max(number(r[f"{kind}_peak_bytes"]) for r in mem) / 2**30,
                                       samples=5, ci_status="not_applicable_maximum", scope="maximum sampled process sum"))
                view = "application" if variant == "AIX" else "provider_dl" if variant in READY[1:] else "provider"
                raw = [r for r in tables["phase_replicates.csv"] if r["model"] == model
                       and CONFIG.get(r["config"], r["config"]) == variant and r["view"] == view]
                bins = {"Inference": ("app_provider_inference",)} if variant == "AIX" else TERRAIN_BINS if variant in READY[1:] else SS_BINS
                for label, names in bins.items():
                    vals = []
                    for rep in range(1, 6):
                        rows = [r for r in raw if int(r["rep"]) == rep]
                        chosen = []
                        for name in names:
                            match = [r for r in rows if r["phase"] == name]
                            if len(match) != 1:
                                raise ValueError(f"Missing terrain phase {base} {rep} {name}")
                            chosen += match
                        for r in chosen:
                            if int(r["steady_steps"]) != 10 or int(r["rank_count"]) != 96 or int(r["denominator"]) != 960:
                                raise ValueError("Terrain phase scope count")
                        vals.append(sum(number(r["ms_per_step"]) for r in chosen))
                    phases.append(dict(base, phase=label, mean_ms=float(np.mean(vals)), ci_half_ms=None,
                                       ci_status="not_shown_scope_comparison", status="recorded", frames=10,
                                       independent_runs=5, scope=chosen[0]["scope"], nodes=";".join(names), samples_ms=json.dumps(vals)))
            else:
                calls = [r for r in tables["latency_calls.csv"] if r["workload"] == model and r["variant"] == variant]
                if len(calls) != 7 or len({r["job"] for r in calls}) != 1:
                    raise ValueError(f"MAIA timing schedule {base}")
                steady = [r for r in calls if r["steady"] == "True"]
                if len(steady) != 5 or [int(r["step"]) for r in steady] != [73, 102, 131, 160, 189]:
                    raise ValueError("MAIA steady schedule")
                values = [number(r["ml_ms"]) for r in steady]
                if any(number(r["ml_ms"]) != number(r["rank_max_ms"]) for r in calls):
                    raise ValueError("MAIA ML step is not actual rank maximum")
                mean, ci, status = mean_ci(values, independent=False)
                latency.append(dict(base, mean_ms=mean, ci_half_ms=ci, ci_status=status, independent_runs=1,
                                    calls_per_run=5, samples_ms=json.dumps(values), scope="plain steady solver-rank maximum ML step"))
                mem = [r for r in tables["memory.csv"] if r["workload"] == model and r["variant"] == variant]
                if len(mem) != 1:
                    raise ValueError("MAIA memory count")
                memory.append(dict(base, metric="Kernel", value_GiB=number(mem[0]["peak_bytes"]) / 2**30,
                                   samples=1, ci_status="not_applicable_maximum", scope=mem[0]["scope"]))
                scope = "dl_whole_client" if variant in READY[1:] else "solver_whole_run"
                regions = [r for r in tables["profile_regions.csv"] if r["workload"] == model
                           and r["variant"] == variant and r["scope"] == scope]
                ranks = [r for r in tables["profile_rank_metrics.csv"] if r["workload"] == model
                         and r["variant"] == variant and r["scope"] == scope]
                parents = {r["cnode"]: r["parent"] for r in ranks}
                bins = {"Inference": ("inference",)} if variant == "AIX" else DL_NODES if variant in READY[1:] else SS_BINS
                selected_nodes = []
                for label, names in bins.items():
                    if variant not in READY:
                        # detailed_regions=false: absence is not a measured zero or MPI proxy.
                        if any(r["region"] in {"smartsim_put_tensor", "smartsim_run_model", "smartsim_unpack_tensor"} for r in regions):
                            raise ValueError("SmartSim instrumentation unexpectedly changed; re-audit")
                        phases.append(dict(base, phase=label, mean_ms=None, ci_half_ms=None,
                                           ci_status="unavailable_not_recorded", status="unavailable_not_recorded",
                                           frames=None, independent_runs=1, scope=scope, nodes=";".join(names), samples_ms="[]"))
                        continue
                    frames = 7 if variant == "AIX" else 6
                    value = inclusive_phase(regions, names, frames, parents)
                    for name in names:
                        row = next(r for r in regions if r["region"] == name)
                        node_ranks = [r for r in ranks if r["cnode"] == row["cnode"]]
                        if len(node_ranks) != 96 or len({r["rank"] for r in node_ranks}) != 96:
                            raise ValueError(f"Region rank count {base} {name}")
                        if any(number(r["visits"]) != frames for r in node_ranks):
                            raise ValueError(f"Per-rank visits mismatch {base} {name}")
                        if not math.isclose(np.mean([number(r["inclusive_s"]) for r in node_ranks]), number(row["inclusive_rankavg_s"]), rel_tol=1e-9):
                            raise ValueError("Region mean mismatch")
                        selected_nodes.append(row["cnode"])
                        audit.append(dict(base, scope=scope, phase=label, region=name, cnode=row["cnode"],
                                          rank_count=96, visits_min=frames, visits_max=frames,
                                          visits_rankavg=number(row["visits_rankavg"]), denominator_frames=frames))
                    phases.append(dict(base, phase=label, mean_ms=value, ci_half_ms=None,
                                       ci_status="unavailable_single_independent_run", status="recorded", frames=frames,
                                       independent_runs=1, scope=scope, nodes=";".join(names), samples_ms=json.dumps([value])))
                assert_disjoint(selected_nodes, parents)
                solver = [r for r in tables["profile_regions.csv"] if r["workload"] == model and r["variant"] == variant and r["scope"] == "solver_whole_run"]
                for name, visits in (("flowex_extract_cubes", 35), ("flowex_reconstruct_output", 7)):
                    match = [r for r in solver if r["region"] == name]
                    if len(match) != 1 or number(match[0]["visits_rankavg"]) != visits:
                        raise ValueError(f"Flowex schedule {base} {name}")
                    row = match[0]
                    rr = [r for r in tables["profile_rank_metrics.csv"] if r["workload"] == model and r["variant"] == variant and r["scope"] == "solver_whole_run" and r["cnode"] == row["cnode"]]
                    if len(rr) != 96 or any(number(r["visits"]) != visits for r in rr):
                        raise ValueError("Flowex per-rank visits")
                    preprocessing.append(dict(base, region=name, visits_rankavg=visits, denominator_calls=7,
                                              amortized_ms=number(row["inclusive_rankavg_s"]) * 1000 / 7,
                                              per_visit_ms=number(row["inclusive_rankavg_s"]) * 1000 / visits,
                                              scope="all seven calls including warmups; extraction has five history updates"))
    hashes = [{"path": str(p.resolve()), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
               "bytes": p.stat().st_size, "rows": len(tables[p.name])} for p in paths]
    return latency, memory, phases, audit, preprocessing, hashes


def style():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                         "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                         "xtick.labelsize": 8, "ytick.labelsize": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42, "savefig.dpi": 170})


def finish_axis(ax, ylabel):
    ax.set_ylabel(ylabel)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#dddddd", linestyle="--", linewidth=.5)
    ax.yaxis.set_major_locator(MaxNLocator(5))
    top = max(ax.get_ylim()[1], ax.dataLim.y1) * 1.28
    ax.set_ylim(0, top)
    decimals = 3 if top < .1 else 2 if top < 1 else 1
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.{decimals}f}"))


def label_bar(ax, x, value, error=0, rotation=0):
    decimals = 6 if 0 < value < .001 else 3 if value < 1 else 2 if value < 10 else 1
    ax.annotate(f"{value:,.{decimals}f}",
                (x, value + error), xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=7, rotation=rotation)


def latency_panel(ax, model, data, variants):
    for x, variant in enumerate(variants):
        row = next(r for r in data if (r["model"], r["variant"]) == (model, variant))
        value, ci = row["mean_ms"] / 1000, row["ci_half_ms"]
        error = ci / 1000 if ci is not None else None
        ax.bar(x, value, width=.58, color=PROVIDER_COLORS[variant], edgecolor="#444444", linewidth=.5,
               yerr=error, capsize=3, error_kw={"linewidth": .8})
        label_bar(ax, x, value, error or 0)
    ax.set_xticks(range(len(variants)), [LABELS[v] for v in variants])
    finish_axis(ax, "Mean ML-step time (s)")


def phase_panel(ax, model, data, variants, readiness=False):
    if readiness:
        labels = list(DL_NODES)
        for j, variant in enumerate(READY[1:]):
            for x, label in enumerate(labels):
                row = next(r for r in data if (r["model"], r["variant"], r["phase"]) == (model, variant, label))
                pos = x + (j - .5) * .36
                ax.bar(pos, row["mean_ms"], width=.34, color=COLORS[label],
                       hatch="///" if j else None, edgecolor="#444444", linewidth=.5)
                label_bar(ax, pos, row["mean_ms"])
        ax.set_xticks(range(4), ["Input\nreceive", "Forward", "Output\nsend", "Other"])
        ax.legend(handles=[Patch(facecolor="white", edgecolor="#444444", label="Blocking"),
                           Patch(facecolor="white", edgecolor="#444444", hatch="///", label="Readiness")],
                  loc="upper right", fontsize=7, frameon=False)
    else:
        for x, variant in enumerate(variants):
            rows = [r for r in data if (r["model"], r["variant"]) == (model, variant)]
            if all(r["mean_ms"] is None for r in rows):
                ax.text(x, .025, "N/A", transform=ax.get_xaxis_transform(), ha="center", fontsize=8, color="#777777")
                continue
            width = .76 / len(rows)
            for j, row in enumerate(rows):
                pos = x + (j - (len(rows) - 1) / 2) * width
                ax.bar(pos, row["mean_ms"], width=width * .9, color=COLORS[row["phase"]],
                       edgecolor="#444444", linewidth=.4)
                label_bar(ax, pos, row["mean_ms"], rotation=90)
        ax.set_xticks(range(len(variants)), [LABELS[v] for v in variants])
    finish_axis(ax, "Mean phase time / scope rank (ms)")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)


def memory_panel(ax, model, data, variants):
    kinds = ["PSS", "RSS"] if model in MODELS[:2] else ["Kernel"]
    for x, variant in enumerate(variants):
        for j, kind in enumerate(kinds):
            row = next(r for r in data if (r["model"], r["variant"], r["metric"]) == (model, variant, kind))
            pos = x + (j - (len(kinds) - 1) / 2) * .32
            ax.bar(pos, row["value_GiB"], width=.3 if len(kinds) == 2 else .58,
                   color=PROVIDER_COLORS[variant], hatch="///" if kind == "RSS" else None,
                   edgecolor="#444444", linewidth=.5)
            label_bar(ax, pos, row["value_GiB"], rotation=90 if len(kinds) == 2 else 0)
    ax.set_xticks(range(len(variants)), [LABELS[v] for v in variants])
    finish_axis(ax, "Maximum memory (GiB)")
    if len(kinds) == 2:
        ax.legend(handles=[Patch(facecolor="white", edgecolor="#444444", label="PSS"),
                           Patch(facecolor="white", edgecolor="#444444", hatch="///", label="RSS")],
                  loc="upper right", fontsize=7, frameon=False)


def phase_legend(fig, full=False):
    fig.legend(handles=[Patch(facecolor=COLORS[k], label=FULL_LEGEND[k] if full else k)
                        for k in FULL_LEGEND], loc="lower center", ncol=2 if full else 4,
               frameon=False, fontsize=8, bbox_to_anchor=(.5, .005), columnspacing=1.8)


def comparison(models, latency, memory, phases, readiness=False):
    variants = READY if readiness else MAIN
    fig, axes = plt.subplots(len(models), 3, figsize=(15.5, 4.6 * len(models) + .6), squeeze=False)
    for row, model in enumerate(models):
        latency_panel(axes[row, 0], model, latency, variants)
        phase_panel(axes[row, 1], model, phases, variants, readiness)
        memory_panel(axes[row, 2], model, memory, variants)
        for ax, title in zip(axes[row], ("(a) ML-step time", "(b) Library inference phases", "(c) Memory footprint")):
            ax.set_title(title, loc="left", pad=12)
        axes[row, 1].text(.5, 1.17, MODEL_LABELS[model], transform=axes[row, 1].transAxes,
                          ha="center", fontsize=13)
    if not readiness:
        phase_legend(fig)
    fig.subplots_adjust(left=.065, right=.985, top=.84 if len(models) == 1 else .91,
                        bottom=.24 if len(models) == 1 else .14, wspace=.32, hspace=.65)
    return fig


def save(fig, output, name, records):
    path = output / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", pad_inches=.12)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=.12)
    with Image.open(path.with_suffix(".png")) as image:
        width, height = image.size
        variation = float(np.asarray(image.convert("RGB")).std())
    if width < 1000 or height < 600 or variation < 8 or path.with_suffix(".pdf").stat().st_size < 1000:
        raise ValueError(f"Invalid rendered image {name}")
    records.append({"name": name, "panels": len(fig.axes), "width": width, "height": height,
                    "pixel_std": variation, "pdf_bytes": path.with_suffix(".pdf").stat().st_size})
    plt.close(fig)


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


NOTES = """## Measurement Notes

Main figures exclude readiness; readiness figures compare AIx reference with blocking and readiness.
All axes are linear, with independent subplot scales. Numeric labels are means or maxima, not medians.
Terrain latency whiskers: 95% t CI over five independent run means, t=2.776445, df=4;
each run mean uses ten steady calls. MAIA latency: actual solver-rank maximum ML-step time,
averaged over five sequential steady calls in one run; independent-run CI unavailable.
No phase or memory CI is drawn. No missing metric is represented by zero: N/A means not recorded.

Library panels are actual instrumented scope times, never scaled to plain latency.
Terrain phases use five replicate means of ten steady steps and the existing SIMPLIFIED phase bins:
AIx application provider inference, PhyDLL DL-side provider scope, SmartSim solver-side client scope.
SmartSim model request is client elapsed time, not isolated RedisAI compute.
MAIA AIx uses inclusive inference / seven visits per solver rank, including warmups.
MAIA PhyDLL uses six recorded frames per DL rank: first frame excluded, one warmup remains;
these are NOT five steady calls. Input combines inclusive dl_recv and dl_frame_copy;
forward uses dl_torch_forward; output uses dl_send_output (never its dl_send descendant again).
Other includes only allocation, unpack, H2D, D2H and reorder staging nodes. Selected inclusive nodes
are checked for ancestor overlap, and per-cnode counts are checked separately on all 96 scope ranks.
SmartSim MAIA detailed client nodes are absent (detailed_regions=false); no MPI/root proxy is used.
See data/region_counts.csv for actual group, node, communicator and visit counts.

Terrain memory is the maximum sampled summed process PSS and RSS across five replicate jobs.
MAIA memory is one kernel whole-job cgroup lifetime high-water mark, including startup and charged
page cache, in GiB. These different measures are not interchangeable or inference-only footprints.
Neither maximum has a CI. Memory runs are separate from timing/profile runs.

Optional preprocessing is explicitly scoped, not whole preprocessing/postprocessing:
flowex_extract_cubes has 35 visits/rank (five history updates x seven calls), reconstruction seven.
Supplementary values amortize inclusive totals / seven calls, including warmups; per-visit values
are also preserved in data/preprocessing.csv. They are not inferred from whole elapsed time.

## Caption Suggestions

Describe latency whiskers as terrain-only 95% independent-run t intervals; MAIA has one run.
Describe library bars as provider-specific rank-average instrumented phases with different call scopes.
Describe memory bars as maxima and explicitly distinguish sampled PSS/RSS from kernel cgroup peaks.
Explain N/A as unrecorded instrumentation, not zero runtime.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--include-mlp-ov0", action="store_true")
    parser.add_argument("--preprocessing", action="store_true")
    args = parser.parse_args()
    latency, memory, phases, audit, preprocessing, hashes = load_data()
    style()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    (output / "data").mkdir(exist_ok=True)
    for name, rows in (("latency", latency), ("memory", memory), ("phases", phases),
                       ("region_counts", audit), ("preprocessing", preprocessing)):
        write_csv(output / "data" / (name + ".csv"), rows)
    records = []
    for model in selected_models(args.include_mlp_ov0):
        folder = "supplement" if model == "mlp_ov0" else "main"
        save(comparison([model], latency, memory, phases), output, f"{folder}/{model}_comparison", records)
        save(comparison([model], latency, memory, phases, True), output, f"readiness/{model}_comparison", records)
    for solver, models in (("terrain", MODELS[:2]), ("maia", MODELS[2:])):
        save(comparison(models, latency, memory, phases), output, f"main/{solver}_overview", records)
    for kind in ("latency", "memory", "phases", "readiness_latency"):
        fig, axes = plt.subplots(2, 2, figsize=(12.5, 10))
        for ax, model in zip(axes.flat, MODELS):
            if kind in ("latency", "readiness_latency"):
                latency_panel(ax, model, latency, READY if kind.startswith("readiness") else MAIN)
            elif kind == "memory":
                memory_panel(ax, model, memory, MAIN)
            else:
                phase_panel(ax, model, phases, MAIN)
            ax.set_title(MODEL_LABELS[model], loc="left", pad=12)
        if kind == "phases":
            phase_legend(fig, full=True)
        fig.subplots_adjust(left=.085, right=.98, top=.95, bottom=.22 if kind == "phases" else .12,
                            wspace=.28, hspace=.55)
        folder = "readiness" if kind.startswith("readiness") else "alternatives"
        save(fig, output, f"{folder}/{kind}_2x2", records)
    if args.preprocessing:
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
        for ax, model in zip(axes, MODELS[2:]):
            for j, region in enumerate(("flowex_extract_cubes", "flowex_reconstruct_output")):
                for x, variant in enumerate(MAIN):
                    row = next(r for r in preprocessing if (r["model"], r["variant"], r["region"]) == (model, variant, region))
                    pos = x + (j - .5) * .32
                    ax.bar(pos, row["amortized_ms"] / 1000, width=.3, color=("#4C78A8", "#54A24B")[j])
                    label_bar(ax, pos, row["amortized_ms"] / 1000, rotation=90)
            ax.set_xticks(range(6), [LABELS[v] for v in MAIN])
            ax.set_title(MODEL_LABELS[model], loc="left")
            finish_axis(ax, "Amortized region time / call (s)")
        fig.legend(handles=[Patch(color="#4C78A8", label="Flowex extraction (five history updates)"),
                            Patch(color="#54A24B", label="Flowex reconstruction")], loc="lower center", ncol=2, frameon=False)
        fig.subplots_adjust(left=.08, right=.98, top=.9, bottom=.28, wspace=.25)
        save(fig, output, "supplement/preprocessing", records)
    manifest = {"source_cells": {"maia": 63, "terrain_timing": 70, "terrain_memory": 70, "terrain_profile": 70},
                "source_hashes": hashes, "figures": records, "main_models": list(MODELS),
                "main_variants": list(MAIN), "readiness_variants": list(READY),
                "t_95_df4": T95, "latency_rows": len(latency), "phase_rows": len(phases),
                "phase_unavailable_rows": sum(r["status"] != "recorded" for r in phases)}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    links = "\n".join(f"- `{r['name']}`: [PNG]({r['name']}.png) | [PDF]({r['name']}.pdf)" for r in records)
    command = "/hpcwork/ro092286/smartsim/CPP-ML-Interface/extern/python/smartsim_cuda-12/bin/python3 plot_cpu_comparison.py"
    flags = (" --include-mlp-ov0" if args.include_mlp_ov0 else "") + (" --preprocessing" if args.preprocessing else "")
    (output / "README.md").write_text("# CPU Figure Index\n\nRegenerate from cpu_benchmark:\n\n`" + command + flags + "`\n\n"
        "Prefer main per-model triptychs for detail, solver overviews for compact comparison.\n"
        "Alternatives offer four-model latency, memory and full-legend phase layouts.\n"
        "Readiness is a separate comparison, not a seventh main provider.\n\n" + links + "\n\n" + NOTES
        + "\n## Audit Files\n\n[Manifest and source SHA256](manifest.json), [latency](data/latency.csv), "
        "[phases](data/phases.csv), [memory](data/memory.csv), [region counts](data/region_counts.csv), "
        "[preprocessing](data/preprocessing.csv). Tables retain unavailable CI/status fields.\n")
    print(f"Rendered {len(records)} PNG/PDF pairs in {output.resolve()}")


if __name__ == "__main__":
    main()
