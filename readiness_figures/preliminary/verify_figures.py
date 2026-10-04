"""Independent numeric/export checks and shared-axis rendering assertions."""
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import statistics
import sys

from PIL import Image

OUTPUT = Path(__file__).resolve().parent
SCRIPT = OUTPUT.parents[1] / 'plot_readiness_preliminary.py'
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location('plotter', SCRIPT)
plotter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plotter)
metadata = json.loads((OUTPUT / 'plot_metadata.json').read_text())
assert hashlib.sha256(SCRIPT.read_bytes()).hexdigest() == metadata['script_sha256']
for path, digest in metadata['sources_sha256'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
sources = {Path(p).stem: Path(p) for p in metadata['sources_sha256']}
runs = plotter.read_csv(sources['runs'])
historical = plotter.read_csv(sources['performance_replicates'])
phases = plotter.read_csv(sources['phases'])
for name, renderer, count in [('step_latency', plotter.render_latency_axis, 6),
                              ('dl_phases', plotter.render_phase_axis, 16)]:
    rows = plotter.read_csv(OUTPUT / f'{name}_summary.csv')
    assert len(rows) == count
    for row in rows:
        values = json.loads(row['replicate_means_ms'])
        assert len(values) == int(row['rep_count']) == 5
        recomputed = []
        for rep in range(1, 6):
            if name == 'dl_phases':
                members = row['members'].split('; ')
                selected = [r for r in phases if r['model'] == row['model']
                            and r['config'] == row['config'] and r['rep'] == str(rep)
                            and r['view'] == 'provider_dl' and r['phase'] in members]
                assert len(selected) == len(members)
                assert all(r['scope'] == 'dl_steady_count_validated'
                           and (r['rank_count'], r['steady_steps'], r['denominator']) == ('96', '10', '960')
                           for r in selected)
                recomputed.append(sum(float(r['ms_per_step']) for r in selected))
            else:
                reference = row['config'] == 'AIx historical'
                selected = [r for r in (historical if reference else runs)
                            if r['model'] == row['model'] and r['rep'] == str(rep)
                            and r['config'] == ('AIx' if reference else row['config'])
                            and r['mode'] == 'performance']
                assert len(selected) == 1 and selected[0]['steady_steps'] == '10'
                recomputed.append(float(selected[0]['steady_mean_ms' if reference else 'mean_step_ms']))
        assert all(math.isclose(a, b, rel_tol=1e-12) for a, b in zip(values, recomputed))
        assert math.isclose(float(row['mean_ms']), statistics.mean(recomputed), rel_tol=1e-12)
        assert math.isclose(float(row['sd_rep_mean_ms']), statistics.stdev(recomputed), rel_tol=1e-12)
        row['mean_ms'], row['sd_rep_mean_ms'] = float(row['mean_ms']), float(row['sd_rep_mean_ms'])
    for model in plotter.MODELS:
        fig, axes = plotter.plt.subplots(1, 2, figsize=(16, 6))
        fig.subplots_adjust(left=.07, right=.98, bottom=.22, top=.83, wspace=.24)
        ax = axes[0]
        renderer(ax, rows, model, model)
        if name == 'step_latency':
            assert ax.get_ylabel() == f'Steady Whole ML-step latency ({"ms" if model == "watercnn" else "s"})'
        fig.canvas.draw()
        assert ax.get_yscale() == 'linear' and ax.get_ylim()[0] == 0
        assert len(ax.patches) == len(ax.texts) == (3 if name == 'step_latency' else 8)
        selected = [r for r in rows if r['model'] == model]
        scale = 1 if model == 'watercnn' else 1000
        for bar, text, row in zip(ax.patches, ax.texts, selected):
            assert text.get_position() == (0, 6) and text.anncoords == 'offset points'
            assert math.isclose(text.xy[0], bar.get_x() + bar.get_width() / 2, abs_tol=1e-12)
            assert math.isclose(text.xy[1], (row['mean_ms'] + row['sd_rep_mean_ms']) / scale)
        boxes = [t.get_window_extent() for t in ax.texts]
        assert not any(a.overlaps(b) for a, b in itertools.combinations(boxes, 2))
        plotter.plt.close(fig)
latency = plotter.read_csv(OUTPUT / 'step_latency_summary.csv')
phase_summary = plotter.read_csv(OUTPUT / 'dl_phases_summary.csv')
for rows in (latency, phase_summary):
    for row in rows:
        for field in ('mean_ms', 'sd_rep_mean_ms'):
            row[field] = float(row[field])
for model in plotter.MODELS:
    fig, axes = plotter.plt.subplots(1, 2, figsize=(16, 6))
    plotter.render_latency_axis(axes[0], latency, model, 'Whole ML-step latency')
    plotter.render_phase_axis(axes[1], phase_summary, model, 'PhyDLL inference phases')
    upper = max(ax.get_ylim()[1] for ax in axes)
    axes[1].sharey(axes[0])
    axes[0].set_ylim(0, upper)
    assert axes[0].get_title() == 'Whole ML-step latency'
    assert axes[0].get_shared_y_axes().joined(*axes)
    assert axes[0].get_ylim() == axes[1].get_ylim()
    assert list(axes[0].get_yticks()) == list(axes[1].get_yticks())
    plotter.plt.close(fig)
for name in metadata['figures']:
    with Image.open(OUTPUT / f'{name}.png') as image:
        assert image.size == (6400, 2400)
        assert all(abs(dpi - 400) < .01 for dpi in image.info['dpi'])
    pdf = (OUTPUT / f'{name}.pdf').read_bytes()
    assert pdf.startswith(b'%PDF') and b'/Subtype /Image' not in pdf
assert not list(OUTPUT.glob('dl_substeps*'))
print('PASS: 6 latency and 16 phase rows independently recomputed from sources; five replicates each;')
print('96 ranks / 10 steps / 960 denominator; source and script hashes; zero-based linear columns;')
print('fixed centered six-point labels without overlaps; four 400-dpi PNGs and vector PDFs; no obsolete artifacts.')
