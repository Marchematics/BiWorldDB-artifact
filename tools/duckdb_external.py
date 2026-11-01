#!/usr/bin/env python3
"""External systems probe: DuckDB full-current-state execution of the declared workload.

This is an *external baseline*, not part of the authored arms. DuckDB holds the genuine
current relation for each version; row edits are real UPDATE statements on the indexed
table, factor edits update the dimension table, and each declared range query is a real
SQL aggregate `SUM(r.amount * s.mul)` with a time-range filter. The whole version-tree
traversal is charged (apply/undo, queries, teardown).

Contract notes
- Same declared workload as the authored workers (see hybrid/common.hpp): 2,049 states,
  absolute edits, per-state nested declared ranges (BWH_RANGE_SCOPE=per_state).
- Answers are exact integers; the complete answer vector is written and can be compared
  with the authored oracle vector.
- Python/DuckDB import and interpreter startup happen before the first timestamp; the
  internal total excludes them. A supervisor should record process wall separately.
"""
import argparse
import hashlib
import json
import struct
import sys
import time

import duckdb
import numpy as np
import pyarrow as pa

import test_reference as ref


class LazyRows:
    """Row accessor compatible with test_reference.make_workload without materializing
    2.3M Python tuples."""

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
    """Edits from the shared reference generator (scope-independent) + per-state ranges."""
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
            order.append(-v - 1)  # exit marker
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
    t = np.zeros(n_rows, dtype=np.int64)
    g = np.zeros(n_rows, dtype=np.int32)
    amt = np.zeros(n_rows, dtype=np.int64)
    ex = np.zeros(n_rows, dtype=np.int8)
    rows = f['_t']
    amts = f['_a']
    grps = f['_g']
    t[:N] = rows
    amt[:N] = amts
    g[:N] = grps
    ex[:N] = 1
    for k in range(n_ins):
        t[N + k] = base['ins_t'][k]
        g[N + k] = base['ins_g'][k]
    mul = np.array(f['factors'], dtype=np.int64)

    con = duckdb.connect()
    con.execute('CREATE TABLE r(id BIGINT, t BIGINT, g INTEGER, amount BIGINT, present TINYINT)')
    tbl = pa.table({'id': np.arange(n_rows, dtype=np.int64), 't': t, 'g': g,
                    'amount': amt, 'present': ex})
    con.register('r_ext', tbl)
    con.execute('INSERT INTO r SELECT * FROM r_ext')
    con.unregister('r_ext')
    con.execute('CREATE INDEX r_id ON r(id)')
    con.execute('CREATE TABLE s(g INTEGER, mul BIGINT)')
    con.register('s_ext', pa.table({'g': np.arange(G, dtype=np.int32), 'mul': mul}))
    con.execute('INSERT INTO s SELECT * FROM s_ext')
    con.unregister('s_ext')
    con.execute('CREATE INDEX s_g ON s(g)')
    # One SQL statement per state for the state's whole declared query bag: a DBMS may
    # legitimately batch the bag, so DuckDB gets one full-state scan per state rather than
    # one per query. This is the strongest practical SQL full-state formulation.
    cols = ", ".join(
        "COALESCE(SUM(CASE WHEN r.t >= ? AND r.t < ? THEN r.amount * s.mul END), 0)"
        for _ in range(args.q))
    q_sql = f"SELECT {cols} FROM r JOIN s ON r.g = s.g WHERE r.present = 1"
    upd = {
        'R': 'UPDATE r SET amount = ?, present = 1 WHERE id = ?',
        'I': 'UPDATE r SET t = ?, g = ?, amount = ?, present = 1 WHERE id = ?',
        'D': 'UPDATE r SET present = 0 WHERE id = ?',
        'M': 'UPDATE r SET t = ?, g = ?, amount = ?, present = 1 WHERE id = ?',
        'U': 'UPDATE r SET t = ?, g = ?, amount = ?, present = ? WHERE id = ?',
        'F': 'UPDATE s SET mul = ? WHERE g = ?',
    }
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
            con.execute(upd['F'], [int(e[2]), gidx])
            mul[gidx] = e[2]
            return
        rid = e[1]
        if record:
            undo_rid[v] = rid
            undo_t[v], undo_g[v], undo_amt[v], undo_ex[v] = t[rid], g[rid], amt[rid], ex[rid]
        if e[0] == 'R':
            con.execute(upd['R'], [int(e[2]), int(rid)])
            amt[rid], ex[rid] = e[2], 1
        elif e[0] == 'I' or e[0] == 'M':
            con.execute(upd['I'], [int(e[2]), int(e[3]), int(e[4]), int(rid)])
            t[rid], g[rid], amt[rid], ex[rid] = e[2], e[3], e[4], 1
        elif e[0] == 'D':
            con.execute(upd['D'], [int(rid)])
            ex[rid] = 0

    def undo(v):
        e = base['edits'][v]
        if e[0] == 'F':
            gidx, old = saved_factor[v]
            con.execute(upd['F'], [old, gidx])
            mul[gidx] = old
            return
        rid = undo_rid[v]
        con.execute(upd['U'], [int(undo_t[v]), int(undo_g[v]), int(undo_amt[v]),
                               int(undo_ex[v]), int(rid)])
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
            params = []
            for k in range(args.q):
                params.extend((int(qs[k][0]), int(qs[k][1])))
            row = con.execute(q_sql, params).fetchone()
            for k in range(args.q):
                out[(v - 1) * args.q + k] = int(row[k])
    t3 = time.monotonic_ns()

    digest = hashlib.sha256(out.tobytes()).hexdigest()
    n_answers = int(out.size)
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

    con.close()
    t5 = time.monotonic_ns()
    rec = {
        'engine': 'duckdb', 'duckdb_version': duckdb.__version__,
        'fixture': args.fixture, 'topology': args.topology, 'seed': args.seed,
        'factor_period': args.factor_period, 'q': args.q, 'repeat': args.repeat,
        'n_answers': n_answers, 'answers_sha256': digest,
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
