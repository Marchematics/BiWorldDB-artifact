#!/usr/bin/env python3
"""Build workers, compare exact answers, and rerun the paper's experiment families."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'tools'))
import joint_planner as features
import joint_planner_fast_v3 as planner

PRODUCTION = dict(BWH_CLASS_LUT='1', BWH_OVERLAY='1', BWH_TRANSPOSE_BASEP='auto',
                  BWH_GROUP_PREFIX='active', BWH_AGG2='auto', BWH_RANGE_SCOPE='per_state')
ADDITIVE = {'shared': ('biworld', ['--arm', 'hybrid']),
            'compiled': ('compiled', []), 'batch': ('batch_serial', []),
            'aligned': ('aligned', ['--arm', 'pgdg_auto']),
            'fenwick': ('classic', ['--arm', 'rf'])}


def run(command, **kw):
    p = subprocess.run([str(x) for x in command], cwd=ROOT, text=True,
                       capture_output=True, **kw)
    if p.returncode:
        raise RuntimeError(f'{Path(str(command[0])).name}: exit {p.returncode}\n{p.stderr[-3000:]}\n{p.stdout[-1000:]}')
    return p.stdout


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def build(states, targets, jobs, maxq=32):
    directory = ROOT / 'build' / (str(states) if maxq == 32 else f'{states}_q{maxq}')
    run(['cmake', '-S', '.', '-B', directory, '-DCMAKE_BUILD_TYPE=Release',
         f'-DBW_STATES={states}', f'-DBW_MAXQ={maxq}'])
    print(f'Building {len(targets)} workers ({states} states)', flush=True)
    run(['cmake', '--build', directory, '-j', jobs, '--target', *sorted(set(targets))])
    return directory / 'bin'


def environment(extra=None, threads=1):
    env = {k: v for k, v in os.environ.items() if not k.startswith('BWH_')}
    env.update(PRODUCTION)
    env['OMP_NUM_THREADS'] = str(threads)
    if extra:
        env.update(extra)
    return env


def synthetic(name, rows, groups, seed=20261131, span=86400000000):
    path = ROOT / 'fixtures' / f'{name}.bin'
    run([sys.executable, 'tools/prep_scale_workloads.py', 'gen', '--rows', rows,
         '--groups', groups, '--span', span, '--seed', seed, '--out', path])
    return path


def fixture_args(fixture, q=2, fp=37, seed=7, topo=0):
    return ['--fixture', fixture, '--base-id', fixture.stem, '--topology', topo,
            '--seed', seed, '--factor-period', fp, '--q', q]


def generated_args(rows, groups, states, q=2, fp=37, seed=7, topo=0):
    return ['--rows', rows, '--groups', groups, '--states', states, '--q', q,
            '--factor-period', fp, '--seed', seed, '--topology', topo]


def worker(bins, binary, arguments, directory, name, repeat=0, env=None):
    directory.mkdir(parents=True, exist_ok=True)
    ans, rec = directory / f'{name}.bin', directory / f'{name}.json'
    start = time.perf_counter_ns()
    run([bins / binary, *arguments, '--repeat', repeat, '--answers', ans, '--json', rec],
        env=env or environment(), timeout=3600)
    process_ns = time.perf_counter_ns() - start
    data = json.loads(rec.read_text())
    data.update(answers_sha256=digest(ans), answer_bytes=ans.stat().st_size,
                process_ns=process_ns)
    # Published and generated metadata use paths relative to the checkout.
    def relative(v):
        if isinstance(v, dict): return {k: relative(x) for k, x in v.items()}
        if isinstance(v, list): return [relative(x) for x in v]
        if isinstance(v, str): return v.replace(str(ROOT) + '/', '')
        return v
    data = relative(data)
    dump(rec, data)
    return data


def oracle(bins, fixture, directory, q=2, fp=37, seed=7, topo=0):
    path = directory / 'oracle.bin'
    directory.mkdir(parents=True, exist_ok=True)
    args = fixture_args(fixture, q, fp, seed, topo)
    # The independent scan has a smaller argument set than timing workers.
    args = args[:2] + args[4:]
    run([bins / 'oracle', *args, '--out', path], env=environment())
    return digest(path), path.stat().st_size


def exact(records, expected=None):
    signatures = {(x['answers_sha256'], x['answer_bytes']) for x in records}
    if expected is not None: signatures.add(expected)
    if len(signatures) != 1:
        raise RuntimeError('Complete answer vectors differ')
    if next(iter(signatures))[1] == 0:
        raise RuntimeError('Empty answer vector')


def summarize(records):
    times = [r['full_ns'] / 1e6 for r in records]
    rng = random.Random(20261102)
    boot = sorted(statistics.median(rng.choices(times, k=len(times))) for _ in range(4000))
    return dict(full_ms=statistics.median(times), median_ci95_ms=[boot[100], boot[3899]],
                process_ms=statistics.median(r['process_ns'] for r in records) / 1e6,
                rss_mib=statistics.median(r.get('peak_rss_kb', 0) for r in records) / 1024,
                answers_sha256=records[0]['answers_sha256'], repetitions=len(records))


def compare(bins, candidates, args, directory, reps, envs=None, expected=None, threads=1):
    records = {name: [] for name in candidates}
    names = list(candidates)
    for rep in range(reps):
        order = names[rep % len(names):] + names[:rep % len(names)]
        for name in order:
            binary, extra = candidates[name]
            records[name].append(worker(bins, binary, [*args, *extra], directory,
                                        f'{name}_r{rep}', rep,
                                        environment((envs or {}).get(name), threads)))
    exact([x for rs in records.values() for x in rs], expected)
    return {name: summarize(rs) for name, rs in records.items()}


def check(args):
    targets = ['biworld', 'compiled', 'batch_serial', 'aligned', 'classic', 'sparse',
               'oracle', 'rectangles', 'rectangles_product', 'holistic', 'adhoc', 'adhoc_oracle', 'fuzz']
    bins = build(2049, targets, args.jobs)
    out = ROOT / 'out/check'
    fixture = synthetic('check', 2000, 4, seed=20261131)
    cases = {}
    for topo in (0, 1, 2):
        for fp in (4, 37):
            directory = out / f'additive_t{topo}_f{fp}'
            expected = oracle(bins, fixture, directory, fp=fp, topo=topo)
            candidates = dict(ADDITIVE, csr=('sparse', ['--arm', 'sparse']))
            cases[directory.name] = compare(bins, candidates,
                fixture_args(fixture, fp=fp, topo=topo), directory, 1,
                envs={'compiled': {'BWH_BATCH_FORCE': '1'}, 'csr': {'BWH_SPARSE_BASE': '1'}},
                expected=expected)
    for topo in (0, 1):
        cases[f'rectangles_t{topo}'] = compare(bins,
            {m: ('rectangles', ['--mode', m]) for m in ('fmd', 'sparse2d', 'densefen', 'oracle')},
            generated_args(400, 4, 50, topo=topo) + ['--c0', 16, '--c1', 16],
            out / f'rectangles_t{topo}', 1)
        candidates = {m + '_' + v: ('holistic', ['--mode', m, '--variant', v])
                      for m in ('hros', 'fenwick') for v in ('eager', 'skip', 'lazy')}
        candidates['oracle'] = ('holistic', ['--mode', 'oracle'])
        cases[f'holistic_t{topo}'] = compare(bins, candidates,
            generated_args(400, 4, 50, fp=2, topo=topo) + ['--burst', 4],
            out / f'holistic_t{topo}', 1)
    for fraction in (20, 50, 100):
        cases[f'adhoc_{fraction}'] = compare(bins,
            {'raw_rank': ('adhoc', ['--arm', 'deri']), 'oracle': ('adhoc_oracle', [])},
            fixture_args(fixture) + ['--adhoc-fraction', fraction], out / f'adhoc_{fraction}', 1)
    run([bins / 'fuzz', '--cases', 300, '--json', out / 'branching.json'])
    fuzz = json.loads((out / 'branching.json').read_text())
    if not fuzz.get('pass', fuzz.get('all_exact', False)):
        raise RuntimeError('Randomized branching check failed')
    dump(out / 'summary.json', {'status': 'PASS', 'cases': cases, 'branching': fuzz})
    print(f'PASS: {len(cases)} complete-vector comparisons and 300 branching cases')


def additive(args, out):
    bins = build(args.states, [v[0] for v in ADDITIVE.values()] + ['oracle', 'aligned_parallel'], args.jobs)
    fixture = args.fixture or synthetic('additive', 100000, 64)
    summary = {}
    qs = (2,) if args.small else (2, 8, 32)
    for topo in (0, 1, 2):
        for seed in ((7,) if args.small else (7, 17, 29)):
            for fp in ((37,) if args.small else (4, 37)):
                for q in qs:
                    key = f't{topo}_s{seed}_f{fp}_q{q}'
                    directory = out / key
                    expected = oracle(bins, fixture, directory, q, fp, seed, topo)
                    candidates = dict(ADDITIVE)
                    if args.threads > 1: candidates['aligned_parallel'] = ('aligned_parallel', ['--arm', 'pgdg_auto'])
                    summary[key] = compare(bins, candidates, fixture_args(fixture, q, fp, seed, topo),
                                           directory, args.repeats, expected=expected, threads=args.threads)
    return summary


def compiler(args, out):
    bins = build(args.states, ['biworld', 'compiled', 'compiled_no_carry', 'batch_parallel', 'aligned', 'aligned_parallel'], args.jobs)
    fixture = args.fixture or synthetic('compiler', 100000, 64)
    candidates = {'compiled': ('compiled', []), 'shared_parallel': ('batch_parallel', []),
                  'aligned_serial': ('aligned', ['--arm', 'pgdg_auto']),
                  'aligned_parallel': ('aligned_parallel', ['--arm', 'pgdg_auto'])}
    envs = {'compiled': {'BWH_BATCH_FORCE': '1'}}
    summary = compare(bins, candidates, fixture_args(fixture), out / 'comparison',
                      args.repeats, envs=envs, threads=args.threads)
    if args.states >= 8192:
        candidates = {'without_carry': ('compiled_no_carry', [])}
        envs = {}
        for blocks in (1, 8, 32):
            key = f'blocks_{blocks}'
            candidates[key] = ('compiled', [])
            envs[key] = {'BWH_V7_BLOCKS': str(blocks), 'BWH_BATCH_FORCE': '1'}
        summary['carry'] = compare(bins, candidates, fixture_args(fixture), out / 'carry',
                                   args.repeats, envs=envs, threads=args.threads)
    return summary


def holistic(args, out):
    bins = build(2049, ['holistic'], args.jobs)
    cells = json.loads((ROOT / 'policies/cells.json').read_text())['holistic']
    if args.small: cells = [('small', 400, 4, 50, 2, 2, 1, 4, ('hros', 'fenwick', 'oracle'))]
    summary = {}
    for name, rows, groups, states, q, fp, topo, burst, modes in cells:
        candidates = {m + '_' + v: ('holistic', ['--mode', m, '--variant', v])
                      for m in modes if m != 'oracle' for v in ('eager', 'skip', 'lazy')}
        if 'oracle' in modes: candidates['oracle'] = ('holistic', ['--mode', 'oracle'])
        summary[name] = compare(bins, candidates,
            generated_args(rows, groups, states, q, fp, topo=topo) + ['--burst', burst],
            out / name, args.repeats)
    return summary


def rectangles(args, out):
    bins = build(2049, ['rectangles'], args.jobs)
    summary = {}
    for domain in ((16,) if args.small else (1024, 4096, 8192)):
        summary[str(domain)] = compare(bins,
            {m: ('rectangles', ['--mode', m]) for m in ('fmd', 'sparse2d', 'densefen')},
            generated_args(400 if args.small else 1000000, 4 if args.small else 64,
                           50 if args.small else 300000) + ['--c0', domain, '--c1', domain],
            out / str(domain), args.repeats)
    return summary


def mixed(args, out):
    bins = build(2049, ['biworld', 'compiled', 'aligned', 'classic', 'batch_serial',
        'rectangles_product', 'rectangles_aligned', 'holistic_reference', 'adhoc', 'adhoc_oracle'], args.jobs)
    fixture = synthetic('mixed100k', 100000, 64)
    cells = json.loads((ROOT / 'policies/cells.json').read_text())['mixed']
    policy = json.loads((ROOT / 'policies/additive.json').read_text())
    mapping = {'panel': 'biworld', 'panel_classic': 'classic', 'panel_batch_v2': 'batch_serial',
               'panel_pgdg_align': 'aligned', 'panel_fmd': 'rectangles_product',
               'panel_fmd_align': 'rectangles_aligned', 'panel_hros_factor': 'holistic_reference',
               'panel_deri': 'adhoc'}
    summary = {}
    for c in cells:
        directory = out / c['cid']
        if c['cls'] in ('1d_declared', 'adhoc_1d'):
            base_args = fixture_args(fixture, c['q'], c['fp'], c['seed'], c['topo'])
        else:
            base_args = generated_args(c['rows'], c['groups'], c['states'], c['q'], c['fp'], c['seed'], c['topo'])
            if c['cls'] == '2d_rectangle': base_args += ['--c0', c['c0'], '--c1', c['c1']]
        candidates = {name: (mapping[binary], extra) for name, (binary, extra) in c['backends'].items()}
        envs = {}
        if c['cls'] == '1d_declared':
            candidates['batch_v7'] = ('compiled', [])
            envs['batch_v7'] = {'BWH_BATCH_FORCE': '1'}
            f = features.case_features({'states': 2049, 'q': c['q'], 'factor_period': c['fp']}, fixture)
            selected = {'pgdg_align': 'aligned_pgdg', 'rf': 'rf', 'batch_deep': 'batch_v7',
                        'hybrid_fast': 'batch_v2', 'corr': 'static_declared'}[planner.decide(policy, f)['chosen']]
        elif c['cls'] == 'adhoc_1d':
            candidates.pop('scan')
            candidates['oracle'] = ('adhoc_oracle', ['--adhoc-fraction', c['adhoc']])
            selected = 'deri_union'
        elif c['cls'] == '2d_rectangle':
            selected = 'naive2d' if c['groups'] * c['c0'] * c['c1'] <= 262144 else 'fmd'
        else: selected = 'hros_f'
        measurements = compare(bins, candidates, base_args, directory, args.repeats, envs=envs)
        best = min(x['full_ms'] for x in measurements.values())
        summary[c['cid']] = {'selected': selected, 'selected_over_best': measurements[selected]['full_ms'] / best,
                             'measurements': measurements}
    return summary


def csr(args, out):
    bins = build(2049, ['sparse'], args.jobs, maxq=128)
    candidates = {'csr': ('sparse', ['--arm', 'sparse']),
                  'row_major': ('sparse', ['--arm', 'hybrid']),
                  'transposed': ('sparse', ['--arm', 'hybrid']),
                  'optimized_dense': ('sparse', ['--arm', 'hybrid']),
                  'fenwick': ('sparse', ['--arm', 'full_projected'])}
    envs = {'csr': {'BWH_SPARSE_BASE': '1'},
            'row_major': {'BWH_TRANSPOSE_BASEP': '0', 'BWH_AGG2': '0', 'BWH_CORR_PREFIX': '0'},
            'transposed': {'BWH_TRANSPOSE_BASEP': '1', 'BWH_AGG2': '0', 'BWH_CORR_PREFIX': '0'},
            'optimized_dense': {'BWH_AGG2': 'auto', 'BWH_CORR_PREFIX': 'auto'}}
    if sys.platform != 'linux': raise RuntimeError('Cold-cache CSR measurements require Linux')
    summary = {}
    fixtures = [args.fixture] if args.fixture else [synthetic(f'csr{n}', n, 61 if n == 1000000 else 512,
                                                          seed=20261029, span=86400000000)
                                                  for n in ((10000,) if args.small else (1000000, 10000000))]
    for fixture in fixtures:
        for q in ((2,) if args.small else (32, 128)):
            key = f'{fixture.stem}_q{q}'
            records = {mode: {name: [] for name in candidates} for mode in ('warm', 'cold')}
            names = list(candidates)
            for rep in range(args.repeats):
                for cold in (0, 1):
                    mode = 'cold' if cold else 'warm'
                    offset = (rep + 2 * cold) % len(names)
                    for name in names[offset:] + names[:offset]:
                        binary, extra = candidates[name]
                        rec = worker(bins, binary, fixture_args(fixture, q) + ['--evict', cold, *extra],
                                     out / key / mode, f'{name}_r{rep}', rep, environment(envs.get(name)))
                        records[mode][name].append(rec)
            exact([r for mode in records.values() for rs in mode.values() for r in rs])
            warmcold = {mode: {name: summarize(rs) for name, rs in data.items()} for mode, data in records.items()}
            budgets = {}
            for budget in (16, 32, 64, 128, 256):
                feasible = {name: warmcold['warm'][name]['full_ms'] for name in candidates
                            if max(warmcold[k][name]['rss_mib'] for k in warmcold) - fixture.stat().st_size / 2**20 <= budget}
                budgets[budget] = min(feasible, key=feasible.get) if feasible else None
            summary[key] = dict(measurements=warmcold, budget_mib=budgets)
    return summary


def planning(args, out):
    policy = json.loads((ROOT / 'policies/additive.json').read_text())
    source = ROOT / 'results/joint_planner_fast_v3_20261031/refined/summary_v3_refined.json'
    cases = json.loads(source.read_text())['cases']
    result = planner.evaluate(policy, cases, boot=2000, quiet=True)
    dump(out / 'recorded_evaluation.json', result)
    return result


def history(args, out):
    if args.fixture is None: raise ValueError('history requires --fixture')
    retail = args.fixture.stem.startswith('retail')
    states = 87065 if retail and not args.small else 2049
    bins = build(states, ['real_history_build', 'real_history', 'real_history_compiled', 'real_history_oracle'], args.jobs)
    out.mkdir(parents=True, exist_ok=True)
    base, spec = out / 'base.bin', out / 'history.bin'
    lo = (1008000 if not args.small else 963360) if retail else 14 * 86400 * 10**6
    hi = (1051200 if not args.small else 1008000) if retail else 2678395000001
    run([bins / 'real_history_build', '--source', args.fixture, '--period-lo', lo, '--period-hi', hi,
         '--bucket', 60 if retail else 3600000000, '--factor-period', 37, '--versions', states,
         '--out-base', base, '--out-spec', spec])
    candidates = {a: ('real_history', ['--arm', a]) for a in ('hybrid', 'pgdg_auto', 'rf')}
    candidates['compiled'] = ('real_history_compiled', [])
    return compare(bins, candidates, ['--fixture', base, '--spec', spec], out / 'runs', args.repeats,
                   envs={name: {'BWH_CORR_PREFIX': 'auto'} for name in candidates})


def branch(args, out):
    if args.fixture is None: raise ValueError('branch requires --fixture')
    retail = args.fixture.stem.startswith('retail')
    trunk, count = (16, 32) if args.small else (512, 4096 if retail else 3072)
    states = trunk + count * 3 + 4
    bins = build(states, ['real_branch_build', 'real_branch', 'real_branch_compiled', 'real_branch_oracle'], args.jobs)
    out.mkdir(parents=True, exist_ok=True)
    base, spec = out / 'base.bin', out / 'history.bin'
    run([bins / 'real_branch_build', '--source', args.fixture,
         '--period-lo', 963360 if retail else 864000000000,
         '--trunk-events', trunk, '--branch-a-events', count, '--branch-b-events', count,
         '--q', 2, '--out-base', base, '--out-spec', spec])
    candidates = {a: ('real_branch', ['--arm', a]) for a in ('hybrid', 'pgdg_auto', 'rf')}
    candidates['compiled'] = ('real_branch_compiled', [])
    return compare(bins, candidates, ['--fixture', base, '--spec', spec], out / 'runs', args.repeats,
                   envs={name: {'BWH_CORR_PREFIX': 'auto'} for name in candidates})


def external(args):
    bins = build(2049, ['biworld', 'emit_events', 'oracle'], args.jobs)
    fixture = args.fixture or synthetic('external', 100000, 64)
    out = ROOT / 'out/external'
    out.mkdir(parents=True, exist_ok=True)
    engines = args.engines.split(',')
    supported = {'dd', 'dbsp', 'duckdb', 'chdb', 'polars', 'datafusion'}
    if not set(engines) <= supported: raise ValueError('Unknown external engine')
    for crate in (x + '_probe' for x in engines if x in ('dd', 'dbsp')):
        print(f'Building {crate}', flush=True)
        run(['cargo', 'build', '--locked', '--release', '--manifest-path', f'external/{crate}/Cargo.toml'])
    summary = {}
    for q in (2, 8, 32):
        directory = out / f'q{q}'
        sha, size = oracle(bins, fixture, directory, q=q)
        events = directory / 'events.bin'
        run([bins / 'emit_events', *fixture_args(fixture, q=q), '--out', events], env=environment())
        records = {}
        for engine in engines:
            if engine in ('dd', 'dbsp'):
                crate = engine + '_probe'
                command = [ROOT / f'external/{crate}/target/release/{crate}', '--events', events,
                           '--q', q, '--oracle', directory / 'oracle.bin']
            else:
                command = [sys.executable, f'tools/{engine}_external.py', '--fixture', fixture,
                           '--topology', 0, '--seed', 7, '--factor-period', 37, '--q', q,
                           '--oracle', directory / 'oracle.bin']
            text = run(command)
            rec = json.loads(text.strip().splitlines()[-1])
            if rec.get('oracle_ok') is not True: raise RuntimeError(f'{engine}: oracle mismatch')
            records[engine] = rec
        records['shared'] = worker(bins, 'biworld', fixture_args(fixture, q=q) + ['--arm', 'hybrid'],
                                    directory, 'shared')
        exact([records['shared']], (sha, size))
        summary[str(q)] = records
    dump(out / 'summary.json', summary)
    print('PASS: external complete-answer comparisons')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    ck = sub.add_parser('check')
    ck.add_argument('--jobs', type=int, default=2)
    bm = sub.add_parser('benchmark')
    bm.add_argument('suite', choices=['additive', 'compiler', 'mixed', 'holistic', 'rectangles', 'csr', 'planning', 'history', 'branch'])
    bm.add_argument('--fixture', type=Path)
    bm.add_argument('--states', type=int, default=2049)
    bm.add_argument('--threads', type=int, default=1)
    bm.add_argument('--repeats', type=int, default=5)
    bm.add_argument('--jobs', type=int, default=2)
    bm.add_argument('--small', action='store_true')
    ex = sub.add_parser('external')
    ex.add_argument('--fixture', type=Path)
    ex.add_argument('--jobs', type=int, default=2)
    ex.add_argument('--engines', default='dd,dbsp', help='dd,dbsp,duckdb,chdb,polars,datafusion')
    args = ap.parse_args()
    if getattr(args, 'fixture', None): args.fixture = args.fixture.resolve()
    if args.command == 'check': return check(args)
    if args.command == 'external': return external(args)
    out = ROOT / 'out' / args.suite
    result = globals()[args.suite](args, out)
    dump(out / 'summary.json', {'system': platform.system(), 'machine': platform.machine(),
                               'suite': args.suite, 'threads': args.threads, 'cases': result})
    print(f'PASS: {args.suite}; results in out/{args.suite}/summary.json')


if __name__ == '__main__':
    main()
