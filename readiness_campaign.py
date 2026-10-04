#!/usr/bin/env python3
"""Isolated paired native-readiness campaign. No production artifacts are modified."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
MINI = BASE / "mini_app"
CPP = BASE / "CPP-ML-Interface"
sys.path.insert(0, str(MINI))
from terrain_cpu_diagnostics import timings, write_csv


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def run(args):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.dataset) or not 0 <= args.task < 10:
        raise ValueError('Invalid dataset/task')
    job = os.environ['SLURM_JOB_ID']
    kind = 'native' if args.mode in ('native', 'yield_smoke') else 'scorep'
    build = ROOT / 'readiness_build' / ('yield_native' if args.mode == 'yield_smoke' else kind)
    solver = build / 'terrain_solver'
    dl = build / 'cpp-ml-interface-build/dl_clients/phydll_dl_client'
    out = ROOT / 'readiness_results' / args.dataset / args.mode / f'job_{job}_{args.task}'
    out.mkdir(parents=True, exist_ok=False)
    libraries = {}
    resolved = {}
    for name, binary in (('solver', solver), ('dl', dl)):
        text = subprocess.check_output(['ldd', str(binary)], text=True)
        (out / f'{name}_libraries.txt').write_text(text)
        if 'not found' in text or (kind == 'native' and 'libscorep' in text):
            raise ValueError('Invalid loaded libraries')
        libraries[name] = {m[0]: str(Path(m[1]).resolve()) for m in
                           re.findall(r'^\s*(\S+)\s+=>\s+(\S+)', text, re.M)}
        resolved.update(libraries[name])
    for lib in ('libphydll.so', 'libtorch_cpu.so'):
        if libraries['solver'][lib] != libraries['dl'][lib]:
            raise ValueError(f'MPMD mismatch: {lib}')
    files = [solver, dl, build / 'generated_registry.hpp', build / 'CMakeCache.txt',
             ROOT / 'readiness_false.toml', ROOT / 'readiness_true.toml', Path(__file__),
             ROOT / 'run_readiness_campaign.sbatch', ROOT / 'build_readiness_campaign.sh',
             ROOT / 'forward_diag_pin.sh', ROOT / 'forward_cpu_diagnostic.hpp',
             MINI / 'terrain_cpu_final.sbatch', MINI / 'terrain_memory_sampler.py',
             MINI / 'terrain_cpu_diagnostics.py', MINI / 'extract_scorep_tables.py']
    files += list((CPP / 'include').rglob('*.hpp')) + list((CPP / 'dl_clients').glob('*.cpp'))
    files += [MINI / 'solver_cpp/terrain_solver.cpp'] + list((MINI / 'solver_cpp/include').rglob('*.hpp'))
    files += [Path(p) for p in resolved.values() if Path(p).is_file()]
    models = ('watercnn', 'giant') if args.mode in ('smoke', 'yield_smoke') else ('giant',) if args.mode == 'native' else \
             ('watercnn' if args.task < 5 else 'giant',)
    rep = args.task % 5 + 1
    steps = 22 if args.mode == 'performance' else 18 if args.mode in ('native', 'yield_smoke') else 6
    for model in models:
        files.extend([MINI / f'train_models/model_a/{model}_cpu.pt',
                      MINI / ('input/prep_watercnn_4k.h5' if model == 'watercnn' else 'input/prep_giant_240x192.h5')])
    hashes = {str(p.resolve()): sha(p) for p in files}
    provenance = dict(mode=args.mode, job=job, task=args.task, rep=rep, steps=steps,
                      solver_ranks=96, dl_ranks=96, pinning='world_rank % 96',
                      intra=1, inter=1, layout='flat_contiguous', rank_grid='12x8',
                      batch_size=10000000, timing_sync=1, memory_sampler=args.mode == 'memory',
                      io_logging=0, io_barriers=0, telemetry=0, internal_memory=0,
                      metadata_version=4, metadata_bytes=88, readiness_offset=76,
                      polling_sleep_us=100, notification='before payload send on duplicated control communicator',
                      inputs=hashes, libraries=libraries, environment=dict(os.environ))
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    (out / 'inputs.sha256').write_text(''.join(f'{h}  {p}\n' for p, h in hashes.items()))
    for command, filename in ((['lscpu'], 'lscpu.txt'), (['scontrol', 'show', 'job', job], 'slurm_job.txt')):
        (out / filename).write_text(subprocess.check_output(command, text=True))
    if args.mode == 'yield_smoke':
        for command, filename in ((['mpirun', '--version'], 'openmpi_version.txt'),
                                  (['ompi_info', '--all', '--parsable'], 'openmpi_mca.txt')):
            (out / filename).write_text(subprocess.check_output(command, text=True))
    for name, repo in (('cmi', CPP), ('mini', MINI), ('cpu', ROOT)):
        (out / f'{name}_git_status.txt').write_text(subprocess.check_output(['git', 'status', '--short'], cwd=repo, text=True))
        (out / f'{name}_git_diff.patch').write_text(subprocess.check_output(['git', 'diff'], cwd=repo, text=True))
        (out / f'{name}_git_head.txt').write_text(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True))
    rows = []
    for model in models:
        policies = (False, True) if rep % 2 else (True, False)
        if args.mode == 'native':
            policies = (False, True, True, False)
        if args.mode == 'yield_smoke':
            policies = (False, False, True, True, False, False)
        outputs = []
        for index, ready in enumerate(policies):
            config = 'PHYDLL_CPP_READY' if ready else 'PHYDLL_CPP'
            yield_explicit = args.mode == 'yield_smoke' and index in (0, 5)
            if yield_explicit:
                config = 'PHYDLL_CPP_YIELD1'
            directory = out / model / f'rep{rep}' / (f'{index + 1}_{config}' if args.mode in ('native', 'yield_smoke') else config)
            directory.mkdir(parents=True)
            binding = directory / 'timings'
            binding.mkdir()
            env = dict(os.environ, SCOREP_EXPERIMENT_DIRECTORY=str(directory / 'scorep'), FORWARD_DIAG_DIR=str(binding))
            env.pop('FORWARD_DIAG_PACED_LIBRARY', None)
            env.pop('TERRAIN_MEMORY_TOKEN', None)
            env.pop('LD_PRELOAD', None)
            env.pop('OMPI_MCA_mpi_yield_when_idle', None)
            if args.mode == 'yield_smoke':
                env['FORWARD_DIAG_MPI_PROVENANCE'] = '1'
                if yield_explicit:
                    env['OMPI_MCA_mpi_yield_when_idle'] = '1'
            config_path = ROOT / f'readiness_{str(ready).lower()}.toml'
            if f'solver_readiness_wait = {str(ready).lower()}' not in config_path.read_text():
                raise ValueError('Readiness config mismatch')
            monitor = None
            monitor_log = None
            if args.mode == 'memory':
                env['TERRAIN_MEMORY_TOKEN'] = f'readiness_{job}_{args.task}_{config}'
                monitor_log = (directory / 'memory_monitor.log').open('w')
                monitor = subprocess.Popen([sys.executable, str(MINI / 'terrain_memory_sampler.py'),
                    '--output', str(directory), '--token', env['TERRAIN_MEMORY_TOKEN'], '--job-id', job,
                    '--solver', str(solver), '--dl', str(dl), '--db-port', '0', '--interval', '1.0',
                    '--expected-solver', '96', '--expected-dl', '96', '--expected-db', '0', '--cleanup'],
                    env=env, stdin=subprocess.DEVNULL, stdout=monitor_log, stderr=subprocess.STDOUT)
                for _ in range(100):
                    if (directory / 'memory.ready').exists():
                        break
                    if monitor.poll() is not None:
                        raise RuntimeError('Sampler failed during startup')
                    time.sleep(.1)
                else:
                    raise RuntimeError('Sampler readiness timeout')
            exports = []
            for name in sorted(env):
                if name.startswith(('SCOREP_', 'MLCOUPLING_', 'PHYDLL_', 'SMARTSIM_FLAG_', 'FORWARD_DIAG_')) or name in (
                        'LD_LIBRARY_PATH', 'HDF5_USE_FILE_LOCKING', 'TERRAIN_MEMORY_TOKEN', 'OMPI_MCA_mpi_yield_when_idle'):
                    exports += ['-x', name]
            wrapper = ['bash', str(ROOT / 'forward_diag_pin.sh')]
            command = ['timeout', '--kill-after=30', '3000', 'mpirun', '--bind-to', 'none', '--oversubscribe',
                       *exports, '-n', '96', *wrapper, str(solver), '--device', 'CPU', '--gpus-per-node', '0',
                       '--ml-nodes', '1', '--model-backend', 'TORCH', '--input-hdf5',
                       str(MINI / ('input/prep_watercnn_4k.h5' if model == 'watercnn' else 'input/prep_giant_240x192.h5')),
                       '--steps', str(steps), '--save-every', str(steps), '--save-mode', 'periodic',
                       '--triangular-scale', '1', '--chunk-size', '12', '--io-mode', 'parallel_hdf5',
                       '--mpi-sync-mode', 'none', '--hdf5-xfer-mode', 'independent', '--rank-grid-x', '12',
                       '--rank-grid-z', '8', '--ml-interface', 'cpp', '--ml-batch-size', '10000000',
                       '--model-path', str(MINI / f'train_models/model_a/{model}_cpu.pt'),
                       '--model-io-layout', 'flat_contiguous', '--cpp-ml-config',
                       str(ROOT / f'readiness_{str(ready).lower()}.toml'), '--output-hdf5', str(directory / 'traj.h5'),
                       ':', *exports, '-n', '96', *wrapper, str(dl)]
            (directory / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
            (directory / 'run.env').write_text(''.join(f'{k}={v}\n' for k, v in sorted(env.items())))
            print(f'START {model} rep{rep} {config}', flush=True)
            start = time.monotonic()
            rc = -1
            try:
                with (directory / 'solver.log').open('w') as log:
                    rc = subprocess.call(command, cwd=directory, env=env, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT)
            finally:
                if monitor:
                    (directory / 'memory.stop').touch()
                    monitor_rc = monitor.wait(timeout=120)
                    monitor_log.close()
                    if monitor_rc:
                        rc = monitor_rc
            row = dict(model=model, rep=rep, config=config, mode=args.mode, job=job,
                       readiness_wait=ready, explicit_yield=yield_explicit, memory_sampler=args.mode == 'memory', steps=steps,
                       steady_steps=(steps - 2) // 2, wall_s=time.monotonic() - start,
                       status='SUCCESS' if rc == 0 else f'FAILED_rc_{rc}', directory=str(directory))
            rows.append(row)
            write_csv(out / 'results.csv', rows)
            if rc:
                raise RuntimeError(f'Run failed: {directory}: {rc}')
            timings(directory / 'solver.log', steps)
            for rank in range(192):
                text = (binding / f'binding_{rank}.txt').read_text().rstrip()
                if not text.endswith(f': {rank % 96}'):
                    raise ValueError(f'Incorrect affinity: {text}')
            if args.mode == 'memory':
                validate_memory(directory)
            if kind == 'scorep' and len(list((directory / 'scorep').glob('*.cubex'))) != 1:
                raise ValueError('Expected exactly one Score-P archive')
            outputs.append(directory / 'traj.h5')
            print(f'COMPLETE {model} rep{rep} {config} wall_s={row["wall_s"]:.3f}', flush=True)
        with (out / f'{model}_hdf5_validation.txt').open('w') as log:
            for output in outputs[1:]:
                subprocess.run(['h5diff', '--exclude-path', '/runtime_seconds', '-v', str(outputs[0]), str(output)],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
    (out / 'COMPLETE').write_text('All runs, affinity, numerical outputs and mode-specific coverage validated.\n')


def validate_memory(directory):
    data = json.loads((directory / 'memory_summary.json').read_text())
    if not data['coverage_ok'] or data['live_read_errors'] or data['cleanup_sigkill_pids']:
        raise ValueError(f'Invalid memory coverage: {directory}')
    if data['max_direct_processes']['solver'] != 96 or data['max_direct_processes']['dl'] != 96:
        raise ValueError(f'Expected exactly 96+96 processes: {directory}')
    return data


def analyze(args):
    from extract_scorep_tables import extract, parse_cubex_file, callpath
    root = ROOT / 'readiness_results' / args.dataset
    runs, steady, raw, phases, memory = [], [], [], [], []
    common_hashes = None
    expected = {(model, str(rep), cfg) for model in ('watercnn', 'giant') for rep in range(1, 6)
                for cfg in ('PHYDLL_CPP', 'PHYDLL_CPP_READY')}
    for mode in ('performance', 'memory'):
        found = set()
        manifests = sorted((root / mode).glob('job_*/results.csv'))
        for manifest in manifests:
            if not (manifest.parent / 'COMPLETE').exists():
                raise ValueError(f'Incomplete job: {manifest.parent}')
            provenance = json.loads((manifest.parent / 'provenance.json').read_text())
            # All measurements must use identical artifacts, not just matching filenames.
            artifacts = {p: h for p, h in provenance['inputs'].items()
                         if p.endswith(('.so', '.pt', '.h5', 'terrain_solver', 'phydll_dl_client', 'generated_registry.hpp'))}
            if common_hashes is None:
                common_hashes = artifacts
            else:
                for path in common_hashes.keys() & artifacts.keys():
                    if common_hashes[path] != artifacts[path]:
                        raise ValueError(f'Artifact changed during campaign: {path}')
                common_hashes.update(artifacts)
            for row in csv.DictReader(manifest.open()):
                key = tuple(row[k] for k in ('model', 'rep', 'config'))
                if key in found or row['status'] != 'SUCCESS':
                    raise ValueError(f'Duplicate or failed run: {row}')
                found.add(key)
                directory = Path(row['directory'])
                values = timings(directory / 'solver.log', 22 if mode == 'performance' else 6)
                meta = {k: row[k] for k in ('model', 'rep', 'config', 'mode', 'job')}
                runs.append({**row, 'mean_step_ms': statistics.mean(v['step_ms'] for v in values)})
                if mode == 'memory':
                    data = validate_memory(directory)
                    memory.append({**meta, 'samples': data['samples'], 'max_sweep_s': data['max_sweep_s'],
                                   'coverage_samples': data['simultaneous_coverage_samples'],
                                   'solver_processes': data['max_direct_processes']['solver'],
                                   'dl_processes': data['max_direct_processes']['dl'],
                                   **{f'{role}_{metric}': data['sampled_simultaneous_peaks'][role][metric]
                                      for role in ('solver', 'dl', 'total') for metric in ('rss_bytes', 'pss_bytes')}})
                    continue
                steady.extend({**meta, **v} for v in values)
                cubex, = (directory / 'scorep').glob('*.cubex')
                trees = parse_cubex_file(cubex)
                # Reuse reviewed scope/count validation and disjoint application/DL extraction.
                r, p = extract(trees, {**meta, 'config': 'PHYDLL_CPP', 'cubex': str(cubex)}, 10)
                if {v['scope']: v['rank_count'] for v in p} != {
                        'solver_steady': 96, 'dl_steady_count_validated': 96}:
                    raise ValueError('Invalid MPMD rank scopes')
                raw.extend({**v, 'config': row['config']} for v in r)
                phases.extend({**v, 'config': row['config']} for v in p)
                # A second, separate provider view attributes self time to deepest semantic ancestor.
                # Readiness owns its MPI_Test children, not the parent's inclusive receive time.
                bins = defaultdict(float)
                mappings = {'phydll_send', 'phydll_recv', 'phydll_readiness_wait', 'phydll_prepack', 'phydll_unpack'}
                solver_count = 0
                for nodes in trees:
                    if not any(n.name == 'solver_step_ml_steady' and n.visits for n in nodes.values()):
                        continue
                    solver_count += 1
                    for node in nodes.values():
                        path = callpath(nodes, node)
                        if 'solver_step_ml_steady' not in path or 'app_provider_inference' not in path:
                            continue
                        phase = next((name for name in reversed(path) if name in mappings), 'Other / overhead')
                        bins[phase] += node.excl_time
                if solver_count != 96:
                    raise ValueError('Invalid provider scope')
                provider_total = sum(v['raw_inclusive_s'] for v in r if v['selected_steady']
                                     and v['region'] == 'app_provider_inference')
                if not abs(sum(bins.values()) - provider_total) <= max(1e-5, provider_total * 1e-5):
                    raise ValueError('Disjoint provider phases do not sum to inference region')
                if row['config'] == 'PHYDLL_CPP_READY' and not any(
                        v['region'] == 'phydll_readiness_wait' and v['selected_steady'] and v['raw_calls'] == 960 for v in r):
                    raise ValueError('Missing readiness steady region/count')
                if row['config'] == 'PHYDLL_CPP' and any(
                        v['region'] == 'phydll_readiness_wait' and v['selected_steady'] and v['raw_calls'] for v in r):
                    raise ValueError('Disabled readiness unexpectedly executed')
                for phase in sorted(mappings | {'Other / overhead'}):
                    phases.append({**meta, 'cubex': str(cubex), 'scope': 'solver_steady', 'rank_count': 96,
                                   'steady_steps': 10, 'denominator': 960, 'view': 'provider_solver',
                                   'phase': phase, 'raw_total_s': bins[phase], 'raw_calls': '',
                                   'ms_per_step': bins[phase] * 1000 / 960, 'regions': ''})
        if found != expected:
            raise ValueError(f'{mode} coverage mismatch missing={expected - found}, unexpected={found - expected}')
    out = root / 'analysis'
    out.mkdir(exist_ok=True)
    for name, rows in (('runs', runs), ('steady_steps', steady), ('scorep_callpaths', raw),
                       ('phases', phases), ('memory', memory)):
        write_csv(out / f'{name}.csv', rows)
    summaries = []
    for mode in ('performance', 'memory'):
        for model in ('watercnn', 'giant'):
            for cfg in ('PHYDLL_CPP', 'PHYDLL_CPP_READY'):
                selected = [r for r in runs if (r['mode'], r['model'], r['config']) == (mode, model, cfg)]
                summaries.append(dict(mode=mode, model=model, config=cfg, reps=len(selected),
                    mean_step_ms=statistics.mean(r['mean_step_ms'] for r in selected),
                    sd_rep_mean_ms=statistics.stdev(r['mean_step_ms'] for r in selected)))
    write_csv(out / 'timing_summary.csv', summaries)
    pairs = []
    for model in ('watercnn', 'giant'):
        for rep in range(1, 6):
            a, b = [next(r for r in runs if (r['mode'], r['model'], r['rep'], r['config']) ==
                         ('performance', model, str(rep), cfg)) for cfg in ('PHYDLL_CPP', 'PHYDLL_CPP_READY')]
            pairs.append(dict(model=model, rep=rep, baseline_ms=a['mean_step_ms'], ready_ms=b['mean_step_ms'],
                              ready_minus_baseline_ms=b['mean_step_ms'] - a['mean_step_ms'],
                              ready_over_baseline=b['mean_step_ms'] / a['mean_step_ms']))
    write_csv(out / 'paired.csv', pairs)
    for name, rows, keys, metrics in (
            ('phase_summary', phases, ('model', 'config', 'view', 'scope', 'phase'), ('ms_per_step',)),
            ('memory_summary', memory, ('model', 'config'), ('solver_rss_bytes', 'solver_pss_bytes',
             'dl_rss_bytes', 'dl_pss_bytes', 'total_rss_bytes', 'total_pss_bytes'))):
        groups = defaultdict(list)
        for row in rows:
            groups[tuple(row[k] for k in keys)].append(row)
        grouped = []
        for key, values in sorted(groups.items()):
            grouped.append({**dict(zip(keys, key)), 'reps': len(values),
                            **{f'mean_{metric}': statistics.mean(v[metric] for v in values) for metric in metrics}})
        write_csv(out / f'{name}.csv', grouped)
    print(json.dumps(summaries, indent=2))


def native(args):
    from analyze_forward_cpu_diagnostic import analyze as analyze_forward
    root = ROOT / 'readiness_results' / args.dataset / 'native'
    jobs = list(root.glob('job_*'))
    if len(jobs) != 1 or not (jobs[0] / 'COMPLETE').exists():
        raise ValueError('Expected one completed native diagnostic job')
    summaries = analyze_forward(jobs[0], runs=('giant/rep1/1_PHYDLL_CPP',
        'giant/rep1/2_PHYDLL_CPP_READY', 'giant/rep1/3_PHYDLL_CPP_READY', 'giant/rep1/4_PHYDLL_CPP'))
    write_csv(jobs[0] / 'forward_summary.csv', summaries)


def yield_smoke(args):
    from analyze_forward_cpu_diagnostic import analyze as analyze_forward
    root = ROOT / 'readiness_results' / args.dataset / 'yield_smoke'
    jobs = list(root.glob('job_*'))
    if len(jobs) != 1 or not (jobs[0] / 'COMPLETE').exists():
        raise ValueError('Expected one completed yield smoke job')
    job = jobs[0]
    manifest = list(csv.DictReader((job / 'results.csv').open()))
    summaries, mpi = [], []
    for model in ('watercnn', 'giant'):
        selected = [r for r in manifest if r['model'] == model]
        if len(selected) != 6:
            raise ValueError('Expected six reversed-order smoke runs per model')
        runs = [str(Path(r['directory']).relative_to(job)) for r in selected]
        summary = analyze_forward(job, runs=runs, batch_counts=
                                  {82368: 16, 85536: 32, 86112: 16, 89424: 32}
                                  if model == 'watercnn' else {288: 32, 576: 64})
        # The reused analyzer writes its detailed output per invocation; retain both models.
        (job / 'timings.csv').rename(job / f'{model}_timings.csv')
        (job / 'summary.json').rename(job / f'{model}_summary.json')
        summaries.extend(dict(model=model, **r) for r in summary)
        for row in selected:
            directory = Path(row['directory']) / 'timings'
            for rank in range(192):
                env = dict(line.split('=', 1) for line in (directory / f'environment_{rank}.txt').read_text().splitlines() if '=' in line)
                expected = '1' if row['explicit_yield'] == 'True' else None
                if env.get('OMPI_MCA_mpi_yield_when_idle') != expected or env.get('LD_PRELOAD'):
                    raise ValueError('Incorrect rank yield environment or interposition')
                paths = list(directory.glob(f'rank_{rank}_pid_*.txt'))
                if len(paths) != 1:
                    raise ValueError('Expected one diagnostic file per rank')
                text = paths[0].read_text()
                value = re.search(r'MPI_EFFECTIVE mpi_yield_when_idle=(\d+) source=MPI_T', text)
                if not value or (expected == '1' and value.group(1) != '1'):
                    raise ValueError('Missing or incorrect effective MCA value')
                mpi.append(dict(model=model, run=Path(row['directory']).name, rank=rank,
                                explicit_yield=expected or 'unset', effective_yield=value.group(1)))
    write_csv(job / 'forward_summary.csv', summaries)
    write_csv(job / 'mpi_rank_validation.csv', mpi)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('run', 'analyze', 'native', 'yield_smoke'))
    parser.add_argument('--mode', choices=('smoke', 'native', 'yield_smoke', 'performance', 'memory'), default='smoke')
    parser.add_argument('--dataset', default='readiness_20261003')
    parser.add_argument('--task', type=int, default=0)
    args = parser.parse_args()
    {'run': run, 'analyze': analyze, 'native': native, 'yield_smoke': yield_smoke}[args.action](args)
