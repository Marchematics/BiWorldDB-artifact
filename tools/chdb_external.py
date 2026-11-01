#!/usr/bin/env python3
"""External systems probe: Apache DataFusion full-current-state execution.

Same declared workload and exactness contract as the DuckDB probe, but the SQL engine is
Apache DataFusion (Rust vectorized query engine, Python bindings). The genuine current
relation is held in numpy arrays and registered to DataFusion as an Arrow table for each
state; the state's whole declared query bag is answered by one SQL statement grouping by
group with one conditional sum per query, and the current factor vector is applied in
Python (exactly, integer arithmetic). This is an external full-state baseline, not an
authored arm.
"""
import argparse
import hashlib
import json
import struct
import sys
import time

import numpy as np
import pyarrow as pa
import chdb

import test_reference as ref


class LazyRows:
    def __init__(self, t, a, g):
        self.t, self.a, self.g = t, a, g

    def __getitem__(self, i):
        return (int(self.t[i]), int(self.a[i]), int(self.g[i]))

    def __len__(self):
        return int(self.t.size)


def load_fixture_fast(path):
    with open(path, 'rb') as fh:
        data = fh.read()
    magic, n, groups, unit, lines, missing, tmin, tmax, scale = struct.unpack_from(
        '<8s q i i q q q q q', data, 0)
    assert magic == b'BWBASE02', magic
    rec = np.frombuffer(data, dtype=np.dtype([('t', '<i8'), ('a', '<i8'),
                                              ('g', '<i4'), ('p', '<i4')]),
                        count=int(n), offset=64)
    off = 64 + 24 * int(n)
    factors = np.frombuffer(data, dtype='<i8', count=int(groups), offset=off).astype(np.int64)
    t = rec['t'].astype(np.int64, copy=True)
    a = rec['a'].astype(np.int64, copy=True)
    g = rec['g'].astype(np.int32, copy=True)
    return {'n': int(n), 'groups': int(groups), 'unit': unit, 'lines': int(lines),
            'missing': int(missing), 'tmin': int(tmin), 'tmax': int(tmax),
            'rows': LazyRows(t, a, g), 'factors': [int(x) for x in factors],
            '_t': t, '_a': a, '_g': g}


def make_workload_per_state(f, topology, seed, fp, q):
    base = ref.make_workload(f, topology, seed, fp, q)
    tmin, tmax = f['tmin'], f['tmax']
    queries = [None] * ref.KSTATES
    for v in range(1, ref.KSTATES):
        r = ref.Rng((seed * 1000000007 + topology * 65537 + fp * 131
                     + v * 2654435761 + 17) & ref.MASK)
        queries[v] = ref.gen_nested(r, tmin, tmax)[:q]
    return base, queries


def preorder(parent):
    kids = [[] for _ in range(ref.KSTATES)]
    for v in range(1, ref.KSTATES):
        kids[parent[v]].append(v)
    order = []
    stack = [(0, False)]
    while stack:
        v, exiting = stack.pop()
        if exiting:
            order.append(-v - 1)
            continue
        if v > 0:
            order.append(v)
        stack.append((v, True))
        for c in reversed(kids[v]):
            stack.append((c, False))
    return order


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--fixture', required=True)
    ap.add_argument('--topology', type=int, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--factor-period', type=int, required=True)
    ap.add_argument('--q', type=int, required=True)
    ap.add_argument('--repeat', type=int, default=0)
    ap.add_argument('--answers', default='')
    ap.add_argument('--oracle', default='')
    ap.add_argument('--json', default='')
    args = ap.parse_args(argv)

    t0 = time.monotonic_ns()
    f = load_fixture_fast(args.fixture)
    t1 = time.monotonic_ns()

    base, queries = make_workload_per_state(f, args.topology, args.seed,
                                            args.factor_period, args.q)
    N, G = f['n'], f['groups']
    n_ins = base['n_ins']
    n_rows = N + n_ins
    t = f['_t'].copy()
    g = f['_g'].copy()
    amt = f['_a'].copy()
    ex = np.ones(N, dtype=np.int8)
    if n_ins:
        t = np.concatenate([t, np.array(base['ins_t'], dtype=np.int64)])
        g = np.concatenate([g, np.array(base['ins_g'], dtype=np.int32)])
        amt = np.concatenate([amt, np.zeros(n_ins, dtype=np.int64)])
        ex = np.concatenate([ex, np.zeros(n_ins, dtype=np.int8)])
    mul = np.array(f['factors'], dtype=np.int64)

    def build_state_table():
        return pa.table({
            't': pa.array(t, type=pa.int64()),
            'g': pa.array(g, type=pa.int32()),
            'amount': pa.array(amt, type=pa.int64()),
            'present': pa.array(ex, type=pa.int8()),
        })
    t2 = time.monotonic_ns()

    out = np.zeros((ref.KSTATES - 1) * args.q, dtype=np.int64)
    # Per-node saved state: an ancestor and a descendant can edit the same row, so the
    # undo record must belong to the version node, not to the row id.
    undo_t = np.zeros(ref.KSTATES, dtype=np.int64)
    undo_g = np.zeros(ref.KSTATES, dtype=np.int32)
    undo_amt = np.zeros(ref.KSTATES, dtype=np.int64)
    undo_ex = np.zeros(ref.KSTATES, dtype=np.int8)
    undo_rid = np.zeros(ref.KSTATES, dtype=np.int64)
    saved_factor = {}
    order = preorder(base['parent'])

    def apply(v, record):
        e = base['edits'][v]
        if e[0] == 'F':
            gidx = e[1]
            saved_factor[v] = (gidx, int(mul[gidx]))
            mul[gidx] = e[2]
            return
        rid = e[1]
        if record:
            undo_rid[v] = rid
            undo_t[v], undo_g[v], undo_amt[v], undo_ex[v] = t[rid], g[rid], amt[rid], ex[rid]
        if e[0] == 'R':
            amt[rid], ex[rid] = e[2], 1
        elif e[0] == 'I' or e[0] == 'M':
            t[rid], g[rid], amt[rid], ex[rid] = e[2], e[3], e[4], 1
        elif e[0] == 'D':
            ex[rid] = 0

    def undo(v):
        e = base['edits'][v]
        if e[0] == 'F':
            gidx, old = saved_factor[v]
            mul[gidx] = old
            return
        rid = undo_rid[v]
        t[rid], g[rid], amt[rid], ex[rid] = undo_t[v], undo_g[v], undo_amt[v], undo_ex[v]

    for item in order:
        if item < 0:
            v = -item - 1
            if v > 0:
                undo(v)
            continue
        v = item
        if v > 0:
            apply(v, True)
            qs = queries[v]
            if args.q == 1:
                lo, hi = int(qs[0][0]), int(qs[0][1])
                sql = (f'SELECT g, SUM(CASE WHEN t >= {lo} AND t < {hi} THEN amount '
                       f'ELSE 0 END) AS m0 FROM r WHERE present = 1 GROUP BY g')
            else:
                cols = ', '.join(
                    f'SUM(CASE WHEN t >= {int(qs[k][0])} AND t < {int(qs[k][1])} '
                    f'THEN amount ELSE 0 END) AS m{k}'
                    for k in range(args.q))
                sql = f'SELECT g, {cols} FROM r WHERE present = 1 GROUP BY g'
            tbl = build_state_table()
            sql_engine = sql.replace(' FROM r ', ' FROM Python(tbl) ')
            df = chdb.query(sql_engine, 'DataFrame')
            per_q = [0] * args.q
            gvals = df['g'].tolist()
            for k in range(args.q):
                col = df[f'm{k}'].tolist()
                for idx, gg in enumerate(gvals):
                    vv = col[idx]
                    if vv is not None:
                        per_q[k] += int(mul[int(gg)]) * int(vv)
            for k in range(args.q):
                out[(v - 1) * args.q + k] = per_q[k]
    t3 = time.monotonic_ns()

    digest = hashlib.sha256(out.tobytes()).hexdigest()
    ok = None
    detail = ''
    if args.oracle:
        expected = np.fromfile(args.oracle, dtype='<i8')
        if expected.size != out.size:
            ok = False
            detail = f'oracle length {expected.size} != {out.size}'
        else:
            ok = bool(np.array_equal(expected, out))
            detail = 'exact match' if ok else 'MISMATCH'
        if not ok:
            bad = np.nonzero(expected != out)[0][:5]
            detail += f' first indices {bad.tolist()}'
    if args.answers:
        out.tofile(args.answers)
    t4 = time.monotonic_ns()
    t5 = time.monotonic_ns()
    rec = {
        'engine': 'clickhouse', 'chdb_version': chdb.__version__,
        'fixture': args.fixture, 'topology': args.topology, 'seed': args.seed,
        'factor_period': args.factor_period, 'q': args.q, 'repeat': args.repeat,
        'n_answers': int(out.size), 'answers_sha256': digest,
        'oracle_ok': ok, 'oracle_detail': detail,
        'phases_ns': {'load': t1 - t0, 'setup': t2 - t1, 'execute': t3 - t2,
                      'extract': t4 - t3, 'cleanup': t5 - t4},
        'full_ns': (t1 - t0) + (t2 - t1) + (t3 - t2) + (t4 - t3) + (t5 - t4),
    }
    if args.json:
        with open(args.json + '.tmp', 'w') as fh:
            json.dump(rec, fh, indent=1)
            fh.write('\n')
        import os
        os.replace(args.json + '.tmp', args.json)
    print(json.dumps(rec))
    return 0 if (ok is None or ok) else 1


if __name__ == '__main__':
    sys.exit(main())
