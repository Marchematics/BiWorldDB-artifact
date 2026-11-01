#!/usr/bin/env python3
"""Scale/stress workload prep for the BiWorldDB Best-Paper plan.

Two subcommands:
  gen     write a synthetic BWBASE02 fixture (N rows, G groups, declared span)
  oracle  generate oracle answer files + index for a grid of (topology, seed,
          factor period, q) using an oracle binary compiled with the matching
          compile-time macros (BWH_STATES / BWH_MAXQ) and the matching runtime
          env (BWH_FACTOR_PCT, BWH_RANGE_SCOPE).

The workload (edit tree, queries, factors) is deterministic from
(fixture, topology, seed, factor period, q) and the knobs in common.hpp, so
the oracle binary must be compiled with the same BWH_STATES/BWH_MAXQ and run
with the same BWH_FACTOR_PCT as the engine binary under test.
"""
import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

MAGIC = b'BWBASE02'
HDR = struct.Struct('<8s q i i q q q q q')
ROW = np.dtype([('t', '<i8'), ('a', '<i8'), ('g', '<i4'), ('p', '<i4')])


def gen(args):
    rng = np.random.default_rng(args.seed)
    n = args.rows
    g = args.groups
    span = args.span
    ts = rng.integers(0, span, size=n, dtype=np.int64)
    grp = rng.integers(0, g, size=n, dtype=np.int32)
    # positive cent-granular amounts with a skewed distribution
    amount = (rng.lognormal(mean=5.0, sigma=1.0, size=n) * 100).astype(np.int64) + 1
    rows = np.empty(n, dtype=ROW)
    rows['t'] = ts
    rows['a'] = amount
    rows['g'] = grp
    rows['p'] = 0
    factors = np.full(g, 1000, dtype=np.int64)
    labels = np.arange(g, dtype=np.int32)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'wb') as f:
        f.write(HDR.pack(MAGIC, n, g, 0, n, 0, 0, int(span) - 1, 2))
        f.write(rows.tobytes())
        f.write(factors.tobytes())
        f.write(labels.tobytes())
    meta = {
        'fixture': str(out),
        'rows': int(n), 'groups': int(g), 'span_us': int(span), 'seed': int(args.seed),
        'bytes': out.stat().st_size,
        'sha256': hashlib.sha256(out.read_bytes()).hexdigest(),
        'declared_span': [0, int(span) - 1],
    }
    Path(str(out) + '.meta.json').write_text(json.dumps(meta, indent=1) + '\n')
    print(json.dumps(meta))


def oracle(args):
    fx = Path(args.fixture).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env['BWH_RANGE_SCOPE'] = 'per_state'
    if args.factor_pct is not None:
        env['BWH_FACTOR_PCT'] = str(args.factor_pct)
    index = {}
    for t in [int(x) for x in args.topologies.split(',')]:
        for s in [int(x) for x in args.seeds.split(',')]:
            for fp in [int(x) for x in args.factor_periods.split(',')]:
                for q in [int(x) for x in args.qs.split(',')]:
                    fid = f'{args.base_id}_t{t}_s{s}_f{fp}_q{q}'
                    ans = outdir / f'{fid}.answers.bin'
                    cmd = [args.oracle_bin, '--fixture', str(fx), '--topology', str(t),
                           '--seed', str(s), '--factor-period', str(fp), '--q', str(q),
                           '--out', str(ans)]
                    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
                    if p.returncode != 0:
                        raise SystemExit(f'oracle failed for {fid}: {p.stderr[-800:]}')
                    rec = json.loads(p.stdout.strip().splitlines()[-1])
                    index[fid] = {'answers_path': str(ans), 'sha256': rec['answers_sha256'],
                                  'n_answers': rec['n_answers'], 'factor_pct': args.factor_pct,
                                  'fixture': str(fx), 'topology': t, 'seed': s,
                                  'factor_period': fp, 'q': q}
    (outdir / 'oracle_index.json').write_text(json.dumps(index, indent=1) + '\n')
    print(json.dumps({'oracle_index': str(outdir / 'oracle_index.json'),
                      'cases': len(index)}))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    g = sub.add_parser('gen')
    g.add_argument('--rows', type=int, default=1_000_000)
    g.add_argument('--groups', type=int, default=256)
    g.add_argument('--span', type=int, default=86_400_000_000)  # 24h in microseconds
    g.add_argument('--seed', type=int, default=20261029)
    g.add_argument('--out', required=True)
    g.set_defaults(func=gen)
    o = sub.add_parser('oracle')
    o.add_argument('--fixture', required=True)
    o.add_argument('--base-id', required=True)
    o.add_argument('--oracle-bin', default='bin/oracle')
    o.add_argument('--outdir', required=True)
    o.add_argument('--topologies', default='0,1,2')
    o.add_argument('--seeds', default='7')
    o.add_argument('--factor-periods', default='4,37')
    o.add_argument('--qs', default='2,8,32')
    o.add_argument('--factor-pct', type=int, default=None)
    o.set_defaults(func=oracle)
    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
