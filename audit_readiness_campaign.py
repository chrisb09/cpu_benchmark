#!/usr/bin/env python3
"""Read-only audit of completed datasets; write only a new job-specific report."""
import csv
import json
import os
from pathlib import Path
import re
import statistics
import subprocess

from readiness_campaign import ROOT, sha, timings, validate_memory, write_csv
from extract_scorep_tables import extract, parse_cubex_file


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main():
    output = ROOT / 'readiness_results' / f'audit_{os.environ["SLURM_JOB_ID"]}'
    output.mkdir(exist_ok=False)
    campaign = ROOT / 'readiness_results/readiness_20261003'
    smoke = ROOT / 'readiness_results/yield_smoke_20261003/yield_smoke/job_4706084_0'
    jobs, checked_hashes, artifacts = set(), {}, {}
    counts = {}
    for mode, root in [('performance', campaign / 'performance'), ('memory', campaign / 'memory'),
                       ('smoke', campaign / 'smoke'), ('native', campaign / 'native'),
                       ('yield_smoke', smoke.parent)]:
        found = set()
        manifests = sorted(root.glob('job_*/results.csv'))
        for manifest in manifests:
            jobdir = manifest.parent
            assert (jobdir / 'COMPLETE').is_file(), jobdir
            provenance = json.loads((jobdir / 'provenance.json').read_text())
            assert provenance['solver_ranks'] == provenance['dl_ranks'] == 96
            assert provenance['intra'] == provenance['inter'] == 1
            assert provenance['polling_sleep_us'] == 100
            jobs.add(provenance['job'])
            for path, digest in provenance['inputs'].items():
                if not path.endswith(('.so', '.pt', '.h5', 'terrain_solver', 'phydll_dl_client', 'generated_registry.hpp')):
                    continue
                assert artifacts.setdefault(path, digest) == digest, ('artifact mismatch', path)
                if path not in checked_hashes:
                    checked_hashes[path] = sha(path)
                assert checked_hashes[path] == digest, ('current artifact changed', path)
            for row in rows(manifest):
                assert row['status'] == 'SUCCESS', row
                directory = Path(row['directory'])
                key = (row['model'], row['rep'], row['config'])
                if mode in ('performance', 'memory'):
                    assert key not in found, key
                found.add(key)
                timings(directory / 'solver.log', int(row['steps']))
                environment = dict(line.split('=', 1) for line in (directory / 'run.env').read_text().splitlines() if '=' in line)
                assert not environment.get('LD_PRELOAD')
                for name in ('PHYDLL_IO_LOG', 'PHYDLL_IO_LOG_BARRIERS', 'PHYDLL_TELEMETRY',
                             'MLCOUPLING_MEM_LOG_VERBOSE', 'MLCOUPLING_MEM_LOG_COLLECTIVE'):
                    assert environment[name] == '0', (directory, name)
                for rank in range(192):
                    assert (directory / 'timings' / f'binding_{rank}.txt').read_text().rstrip().endswith(f': {rank % 96}')
                if mode == 'memory':
                    validate_memory(directory)
                if mode == 'performance':
                    cubex, = (directory / 'scorep').glob('*.cubex')
                    raw, phases = extract(parse_cubex_file(cubex), dict(model=row['model'], rep=row['rep'],
                        config='PHYDLL_CPP', mode=mode, job=row['job'], cubex=str(cubex)), 10)
                    assert {p['scope']: p['rank_count'] for p in phases} == {
                        'solver_steady': 96, 'dl_steady_count_validated': 96}
                    ready = [r for r in raw if r['region'] == 'phydll_readiness_wait' and r['selected_steady']]
                    assert sum(r['raw_calls'] for r in ready) == (960 if row['config'] == 'PHYDLL_CPP_READY' else 0)
            for model in {r['model'] for r in rows(manifest)}:
                outputs = [Path(r['directory']) / 'traj.h5' for r in rows(manifest) if r['model'] == model]
                for target in outputs[1:]:
                    subprocess.run(['h5diff', '--exclude-path', '/runtime_seconds', str(outputs[0]), str(target)], check=True)
        counts[mode] = sum(len(rows(m)) for m in manifests)
        if mode in ('performance', 'memory'):
            assert found == {(m, str(r), c) for m in ('watercnn', 'giant') for r in range(1, 6)
                             for c in ('PHYDLL_CPP', 'PHYDLL_CPP_READY')}
    assert counts == dict(performance=20, memory=20, smoke=4, native=4, yield_smoke=12), counts
    mpi = rows(smoke / 'mpi_rank_validation.csv')
    assert len(mpi) == 2304
    assert len({(r['model'], r['run'], r['rank']) for r in mpi}) == 2304
    for row in mpi:
        directory = smoke / row['model'] / 'rep1' / row['run'] / 'timings'
        rank = row['rank']
        environment = dict(line.split('=', 1) for line in (directory / f'environment_{rank}.txt').read_text().splitlines() if '=' in line)
        assert environment.get('OMPI_MCA_mpi_yield_when_idle', 'unset') == row['explicit_yield']
        assert not environment.get('LD_PRELOAD')
        path, = directory.glob(f'rank_{rank}_pid_*.txt')
        text = path.read_text()
        assert f'MPI_EFFECTIVE mpi_yield_when_idle={row["effective_yield"]} source=MPI_T' in text
        assert len(re.findall(r'^TIMING ', text, re.M)) == 9
        if row['explicit_yield'] == '1':
            assert row['effective_yield'] == '1'
    status = subprocess.check_output(['sacct', '-j', ','.join(sorted(jobs)), '-X', '--parsable2',
        '--format=JobID,State,ExitCode,Partition,Elapsed,NodeList'], text=True)
    (output / 'slurm_status.txt').write_text(status)
    accounting = list(csv.DictReader(status.splitlines(), delimiter='|'))
    assert len(accounting) == len(jobs), ('missing accounting', jobs, accounting)
    assert all(r['State'] == 'COMPLETED' and r['ExitCode'] == '0:0' and r['Partition'] == 'c23mm' for r in accounting)
    summary = []
    forward = rows(smoke / 'forward_summary.csv')
    for model in ('watercnn', 'giant'):
        for config in ('PHYDLL_CPP', 'PHYDLL_CPP_YIELD1', 'PHYDLL_CPP_READY'):
            for kind in ('phydll_forward', 'solver_recv'):
                selected = [r for r in forward if r['model'] == model and r['run'].split('_', 1)[1] == config and r['kind'] == kind]
                assert len(selected) == 2 and all(int(r['samples']) == 576 for r in selected)
                summary.append(dict(model=model, config=config, kind=kind, observations=2,
                    **{k: statistics.mean(float(r[k]) for r in selected) for k in ('mean_wall_s', 'mean_thread_s', 'mean_process_s')}))
    write_csv(output / 'yield_summary.csv', summary)
    result = dict(counts=counts, jobs=sorted(jobs), verified_artifacts=len(checked_hashes),
                  mpi_rank_cases=len(mpi), effective_yield_values=sorted({r['effective_yield'] for r in mpi}),
                  numerical_equality='h5diff passed, excluding only /runtime_seconds')
    (output / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    (output / 'COMPLETE').write_text('Read-only audit passed. Original datasets and summaries were not modified.\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
