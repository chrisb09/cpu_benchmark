#!/usr/bin/env python3
"""Validate retained CSVs and render preliminary figures, without reanalysis."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import statistics as stats
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent
MODELS = ('watercnn', 'giant')
CONFIGS = ('PHYDLL_CPP', 'PHYDLL_CPP_READY')
TITLES = ('[TS] WaterCNN', '[TS] Giant MLP')
COLORS = ('#CC79A7', '#009E73')
GROUPS = {
    'Input\nreceive': ['dl_recv / dl_frame_copy'],
    'Tensor\nstaging': ['dl_allocate', 'dl_input_unpack', 'dl_h2d',
                        'dl_d2h', 'dl_output_reorder', 'Other / overhead'],
    'Torch\nforward': ['dl_torch_forward'],
    'Output\nsend': ['dl_send_output'],
}
PHASE_COLORS = ('#4C78A8', '#BAB0AC', '#F58518', '#54A24B')


def read_csv(path):
    # Historical rows retain large serialized runtime-environment provenance.
    csv.field_size_limit(16 * 1024 * 1024)
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def number(value, positive=True):
    value = float(value)
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f'Invalid measurement: {value}')
    return value


def index(rows, fields, expected):
    result = {}
    for row in rows:
        key = tuple(row[f] for f in fields)
        if key in result:
            raise ValueError(f'Duplicate key: {key}')
        result[key] = row
    if set(result) != expected:
        raise ValueError(f'Coverage mismatch: missing={expected-set(result)}, extra={set(result)-expected}')
    return result


def decimal(value, _=None):
    places = 0 if abs(value) >= 1000 else 1 if abs(value) >= 10 else 3 if abs(value) >= .1 else 4 if abs(value) >= .01 else 6
    return f'{value:.{places}f}'.rstrip('0').rstrip('.') if places else f'{value:.0f}'


def save(fig, output, name):
    fig.canvas.draw()
    for ax in fig.axes:
        labels = ax.get_yticklabels() + ax.get_xticklabels()
        if any('e+' in t.get_text() or 'e-' in t.get_text() or '10^' in t.get_text() for t in labels):
            raise ValueError('Scientific axis formatting is forbidden')
    for suffix in ('png', 'pdf'):
        fig.savefig(output / f'{name}.{suffix}', dpi=400, facecolor='white')
    plt.close(fig)


def render_latency_axis(ax, rows, model, title):
    scale = 1 if model == 'watercnn' else 1000
    unit = 'ms' if scale == 1 else 's'
    selected = [r for r in rows if r['model'] == model]
    for x, row, color in zip(range(3), selected, ('#59788C', *COLORS)):
        mean, sd = row['mean_ms'] / scale, row['sd_rep_mean_ms'] / scale
        ax.bar(x, mean, width=.58, color=color, edgecolor='#444444', linewidth=.7)
        ax.errorbar(x, mean, yerr=sd, fmt='none', color='#333333', capsize=3, lw=1)
        ax.annotate(decimal(mean), (x, mean + sd), xytext=(0, 6),
                    textcoords='offset points', ha='center', fontsize=10)
    ax.set(title=title, xlim=(-.6, 2.6),
           ylim=(0, max((r['mean_ms'] + r['sd_rep_mean_ms']) / scale for r in selected) * 1.18),
           ylabel=f'Steady Whole ML-step latency ({unit})')
    ax.set_xticks(range(3), ['AIx\nreference', 'PhyDLL\nDefault', 'PhyDLL\nReadiness'])
    ax.yaxis.set_major_formatter(FuncFormatter(decimal))
    ax.grid(axis='y', alpha=.18)
    ax.set_axisbelow(True)


def render_phase_axis(ax, rows, model, title):
    scale = 1 if model == 'watercnn' else 1000
    unit = 'ms' if scale == 1 else 's'
    selected = [r for r in rows if r['model'] == model]
    for pi, label in enumerate(GROUPS):
        for ci, cfg in enumerate(CONFIGS):
            row = next(r for r in selected if r['config'] == cfg and r['phase'] == label.replace('\n', ' '))
            mean, sd = row['mean_ms'] / scale, row['sd_rep_mean_ms'] / scale
            x = pi + (ci - .5) * .44
            ax.bar(x, mean, width=.38, color=PHASE_COLORS[pi], hatch='///' if ci else '',
                   edgecolor='#444444', linewidth=.7)
            ax.errorbar(x, mean, yerr=sd, fmt='none', color='#333333', capsize=3, lw=1)
            ax.annotate(decimal(mean), (x, mean + sd), xytext=(0, 6),
                        textcoords='offset points', ha='center', fontsize=9)
    ax.set(title=title, xlim=(-.6, 3.6),
           ylim=(0, max((r['mean_ms'] + r['sd_rep_mean_ms']) / scale for r in selected) * 1.18),
           ylabel=f'DL-rank mean time ({unit})')
    ax.set_xticks(range(4), list(GROUPS))
    ax.yaxis.set_major_formatter(FuncFormatter(decimal))
    ax.grid(axis='y', alpha=.18)
    ax.set_axisbelow(True)


def plot_legend(fig, phases=False):
    handles = []
    if phases:
        handles = [Patch(facecolor='#dddddd', edgecolor='#444444', label='PhyDLL Default'),
                   Patch(facecolor='#dddddd', edgecolor='#444444', hatch='///', label='PhyDLL Readiness')]
    handles.append(Line2D([], [], color='#333333', marker='|', ls='', markersize=12, label='Whiskers: SD'))
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .025),
               ncol=len(handles), frameon=False, fontsize=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, default=ROOT / 'readiness_results/readiness_20261003/analysis')
    parser.add_argument('--historical', type=Path, default=ROOT.parent / 'mini_app/campaign_figures/terrain_cpu_final_production_4644312/performance/performance_replicates.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'readiness_figures/preliminary')
    args = parser.parse_args()
    expected = {(m, str(r), c) for m in MODELS for r in range(1, 6) for c in CONFIGS}
    paths = [args.analysis / f'{name}.csv' for name in
             ('paired', 'runs', 'steady_steps', 'phases', 'phase_summary', 'memory', 'memory_summary')]
    sources = paths + [args.historical]
    hashes = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    tables = {p.stem: read_csv(p) for p in paths}
    runs = index([r for r in tables['runs'] if r['mode'] == 'performance'],
                 ('model', 'rep', 'config'), expected)
    pairs = index(tables['paired'], ('model', 'rep'), {(m, str(r)) for m in MODELS for r in range(1, 6)})
    steps = index(tables['steady_steps'], ('model', 'rep', 'config', 'step'),
                  {k + (str(s),) for k in expected for s in range(4, 23, 2)})
    for key, row in runs.items():
        if row['status'] != 'SUCCESS' or row['steady_steps'] != '10':
            raise ValueError(f'Invalid run: {key}')
        values = [number(steps[key + (str(s),)]['step_ms']) for s in range(4, 23, 2)]
        mean = number(row['mean_step_ms'])
        paired = number(pairs[key[:2]]['baseline_ms' if key[2] == CONFIGS[0] else 'ready_ms'])
        if not math.isclose(mean, stats.mean(values), rel_tol=1e-10) or not math.isclose(mean, paired, rel_tol=1e-10):
            raise ValueError(f'Replicate mean mismatch: {key}')
    historical = index([r for r in read_csv(args.historical) if r['config'] == 'AIx'],
                       ('model', 'rep'), {(m, str(r)) for m in MODELS for r in range(1, 6)})
    for key, row in historical.items():
        if row['status'] != 'SUCCESS' or row['mode'] != 'performance' or row['steady_steps'] != '10':
            raise ValueError(f'Invalid historical reference: {key}')
        samples = json.loads(row['steady_samples_ms'])
        if set(samples) != {str(s) for s in range(4, 23, 2)} or not math.isclose(number(row['steady_mean_ms']), stats.mean(map(number, samples.values())), rel_tol=1e-10):
            raise ValueError(f'Historical mean mismatch: {key}')
    phase_names = {p for phases in GROUPS.values() for p in phases}
    phases = index([r for r in tables['phases'] if r['view'] == 'provider_dl'],
                   ('model', 'rep', 'config', 'phase'), {k + (p,) for k in expected for p in phase_names})
    for key, row in phases.items():
        value = number(row['ms_per_step'], positive=False)
        calls = 0 if key[-1] == 'Other / overhead' else 1920 if key[-1] in ('dl_allocate', 'dl_recv / dl_frame_copy') else 960
        if (row['scope'] != 'dl_steady_count_validated' or row['mode'] != 'performance'
                or (int(row['rank_count']), int(row['steady_steps']), int(row['denominator'])) != (96, 10, 960)
                or int(row['raw_calls']) != calls
                or not math.isclose(value, number(row['raw_total_s'], positive=False) * 1000 / 960, abs_tol=1e-10)):
            raise ValueError(f'Invalid DL phase scope/count/normalization: {key}')
    phase_means = index([r for r in tables['phase_summary'] if r['view'] == 'provider_dl'],
                        ('model', 'config', 'phase'),
                        {(m, c, p) for m in MODELS for c in CONFIGS for p in phase_names})
    for (model, cfg, phase), row in phase_means.items():
        mean = stats.mean(float(phases[(model, str(r), cfg, phase)]['ms_per_step']) for r in range(1, 6))
        if row['reps'] != '5' or not math.isclose(mean, number(row['mean_ms_per_step'], positive=False), abs_tol=1e-10):
            raise ValueError('DL phase summary disagrees with replicates')
    memory = index(tables['memory'], ('model', 'rep', 'config'), expected)
    memory_means = index(tables['memory_summary'], ('model', 'config'), {(m, c) for m in MODELS for c in CONFIGS})
    memory_rows = []
    for model in MODELS:
        for cfg in CONFIGS:
            selected = [memory[(model, str(r), cfg)] for r in range(1, 6)]
            if any(r['mode'] != 'memory' or int(r['solver_processes']) != 96 or int(r['dl_processes']) != 96
                   or int(r['coverage_samples']) <= 0 for r in selected):
                raise ValueError('Invalid memory coverage')
            for metric in ('pss', 'rss'):
                values = [number(r[f'total_{metric}_bytes']) / 1073741824 for r in selected]
                original = memory_means[(model, cfg)]
                if original['reps'] != '5' or not math.isclose(stats.mean(values), number(original[f'mean_total_{metric}_bytes']) / 1073741824, rel_tol=1e-10):
                    raise ValueError('Memory summary disagrees with replicates')
                peak = selected[values.index(max(values))]
                memory_rows.append(dict(model=model, config=cfg, metric=metric.upper(), rep_count=5,
                    mean_GiB=f'{stats.mean(values):.6f}', max_GiB=f'{max(values):.6f}',
                    stddev_GiB=f'{stats.stdev(values):.6f}', min_GiB=f'{min(values):.6f}',
                    max_rep=peak['rep'], max_job=peak['job'], max_sweep_s=max(number(r['max_sweep_s']) for r in selected),
                    source=str((args.analysis / 'memory.csv').resolve()),
                    provenance='maximum across per-run simultaneous process-set sampled peaks; non-atomic largest observed, not theoretical maximum'))
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / 'memory_peak_aggregation.csv', memory_rows)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.titlesize': 14,
                         'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    summary = []
    for model in MODELS:
        ref = [number(historical[(model, str(r))]['steady_mean_ms']) for r in range(1, 6)]
        values = [[number(runs[(model, str(r), cfg)]['mean_step_ms']) for r in range(1, 6)] for cfg in CONFIGS]
        for cfg, group in zip(('AIx historical', *CONFIGS), (ref, *values)):
            mean, sd = stats.mean(group), stats.stdev(group)
            summary.append(dict(model=model, config=cfg, mean_ms=mean,
                                sd_rep_mean_ms=sd, rep_count=len(group),
                                replicate_means_ms=json.dumps(group)))
    phase_summary = []
    for model in MODELS:
        for label, members in GROUPS.items():
            for cfg in CONFIGS:
                values = [sum(number(phases[(model, str(r), cfg, p)]['ms_per_step'], positive=False) for p in members) for r in range(1, 6)]
                mean, sd = stats.mean(values), stats.stdev(values)
                number(mean)
                phase_summary.append(dict(model=model, config=cfg, phase=label.replace('\n', ' '),
                    mean_ms=mean, sd_rep_mean_ms=sd, rep_count=len(values),
                    replicate_means_ms=json.dumps(values), members='; '.join(members)))
    for name, heading, renderer, rows in (
            ('step_latency', 'Whole ML-step latency', render_latency_axis, summary),
            ('dl_phases', 'PhyDLL inference phases', render_phase_axis, phase_summary)):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.subplots_adjust(left=.07, right=.98, bottom=.22, top=.83, wspace=.24)
        for ax, model, title in zip(axes, MODELS, TITLES):
            renderer(ax, rows, model, title)
        fig.suptitle(heading, fontsize=16, y=.96)
        plot_legend(fig, phases=name == 'dl_phases')
        save(fig, args.output, name)
    for model, title, name in zip(MODELS, TITLES, ('watercnn_comparison', 'giant_mlp_comparison')):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [1, 1.3]})
        fig.subplots_adjust(left=.07, right=.98, bottom=.22, top=.83, wspace=.26)
        render_latency_axis(axes[0], summary, model, 'Whole ML-step latency')
        render_phase_axis(axes[1], phase_summary, model, 'PhyDLL inference phases')
        upper = max(ax.get_ylim()[1] for ax in axes)
        axes[1].sharey(axes[0])
        axes[0].set_ylim(0, upper)
        fig.suptitle(title, fontsize=16, y=.96)
        plot_legend(fig, phases=True)
        save(fig, args.output, name)
    write_csv(args.output / 'step_latency_summary.csv', summary)
    write_csv(args.output / 'dl_phases_summary.csv', phase_summary)
    # Remove superseded generated artifacts only after their replacements exist.
    for name in ('dl_substeps.png', 'dl_substeps.pdf', 'dl_substeps_summary.csv'):
        (args.output / name).unlink(missing_ok=True)
    for path in sources:
        if hashlib.sha256(path.read_bytes()).hexdigest() != hashes[str(path.resolve())]:
            raise ValueError(f'Source changed during plotting: {path}')
    metadata = dict(generated_utc=datetime.now(timezone.utc).isoformat(), slurm_job_id=os.environ.get('SLURM_JOB_ID'),
        command=[sys.executable, *sys.argv], sources_sha256=hashes,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        matplotlib_version=matplotlib.__version__, output=str(args.output.resolve()), png_dpi=400,
        figures=['step_latency', 'dl_phases', 'watercnn_comparison', 'giant_mlp_comparison'],
        comparison_axes='shared y-axis limits and ticks within each model comparison',
        units={'step_latency': {'watercnn': 'ms', 'giant': 's'}, 'dl_phases': {'watercnn': 'ms', 'giant': 's'}, 'watercnn_comparison': 'ms', 'giant_mlp_comparison': 's', 'summary_csvs': 'ms', 'memory': 'GiB (1073741824 bytes)'},
        latency_statistic='arithmetic mean of five per-replicate arithmetic means of ten steady solver ML steps',
        latency_plot='zero-based linear mean columns of five replicate means per model/configuration; capped sample SD whiskers; no measurement points; mean labels six points above mean plus SD',
        historical_note='AIx production_4644312 is unpaired; original datasets and Score-P settings differ',
        phase_statistic='exclusive disjoint provider_dl bins / 96 ranks / ten steady calls; five replicate means and sample SD',
        phase_plot='zero-based linear grouped columns; group sums per replicate before mean and sample SD; capped SD whiskers, no measurement points; staging includes allocation, input unpack, h2d, d2h, output reorder and residual overhead; not whole-step makespan; no solver readiness time added',
        memory_note='max of per-run simultaneous total sampled peaks, not summed independent role or process maxima; non-atomic observed lower bound',
        validation='PASS: exact five-replicate keys, paired/run/step mean agreement, historical means, DL phase coverage/counts/normalization, memory census, unchanged sources')
    (args.output / 'plot_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(dict(output=str(args.output.resolve()), validation=metadata['validation'], latency=summary, memory=memory_rows), indent=2))


if __name__ == '__main__':
    main()
