#!/usr/bin/env python3
"""Independent Python reference for the 2026-10-25 hybrid panel.

Reproduces the declared workload from hybrid/common.hpp exactly (splitmix64 RNG),
traverses the version tree with a full mutable row state, and answers the declared
queries incrementally (O(q) per state, with a per-group per-query mass matrix for
factor edits). A periodic direct full-scan audit checks the incremental answers.
Used only to bug-check the C++ arms; not part of any timing result.
"""
import struct
import sys

MASK = (1 << 64) - 1
KSTATES = 2049
KMAXQ = 32


class Rng:
    def __init__(self, seed):
        self.s = (seed & MASK) if seed else 0x9E3779B97F4A7C15

    def next(self):
        self.s = (self.s + 0x9E3779B97F4A7C15) & MASK
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        return (z ^ (z >> 31)) & MASK

    def bounded(self, n):
        return self.next() % n if n else 0

    def range(self, lo, hi):
        return lo + (self.next() % (hi - lo)) if hi > lo else lo


def load_fixture(path):
    with open(path, 'rb') as fp:
        data = fp.read()
    magic, n, groups, unit, lines, missing, tmin, tmax, scale = struct.unpack_from('<8s q i i q q q q q', data, 0)
    assert magic == b'BWBASE02', magic
    rows = []
    off = 64
    for _ in range(n):
        t, amount, g, _pad = struct.unpack_from('<qqii', data, off)
        off += 24
        rows.append((t, amount, g))
    factors = list(struct.unpack_from('<%dq' % groups, data, off))
    off += 8 * groups
    labels = list(struct.unpack_from('<%di' % groups, data, off))
    return dict(n=n, groups=groups, unit=unit, lines=lines, missing=missing,
                tmin=tmin, tmax=tmax, rows=rows, factors=factors, labels=labels)


def version_parent(topology, v):
    if v <= 0:
        return -1
    return {0: v - 1, 1: (v - 1) // 2, 2: 0}[topology]


def gen_nested(r, tmin, tmax):
    out = []
    lo, hi = tmin, tmax + 1
    span0 = hi - lo
    for _ in range(KMAXQ):
        span = hi - lo
        if span <= 4096:
            lo = tmin + span0 // 2
            hi = lo + 2048
            span = hi - lo
        a = span * (3 + r.bounded(16)) // 100
        b = span * (3 + r.bounded(16)) // 100
        lo += a
        hi -= b
        if hi - lo < 1024:
            hi = lo + 1024
        out.append((lo, hi))
    return out


def make_workload(f, topology, seed, fp, q):
    N, G = f['n'], f['groups']
    tmin, tmax = f['tmin'], f['tmax']
    qr = Rng((seed * 1000000007 + topology * 65537 + fp * 131 + 17) & MASK)
    queries = gen_nested(qr, tmin, tmax)[:q]
    rng = Rng((seed * 1000003 + topology * 7919 + fp * 104729 + 20261025) & MASK)
    parent = [version_parent(topology, v) for v in range(KSTATES)]
    edits = [None] * KSTATES
    gfactor = [1000] * G
    next_group = 0
    next_ins = 0
    ins_t, ins_g = [], []
    for v in range(1, KSTATES):
        if fp > 0 and v % fp == 0:
            g = next_group % G
            next_group += 1
            cur = gfactor[g]
            nf = 1000 if cur == 950 else (1050 if cur == 1000 else 950)
            gfactor[g] = nf
            edits[v] = ('F', g, nf)
            continue
        roll = rng.bounded(100)
        if roll < 55:
            rid = rng.range(0, N)
            edits[v] = ('R', rid, f['rows'][rid][1] + rng.range(-100000, 100000))
        elif roll < 75:
            rid = N + next_ins
            next_ins += 1
            t = rng.range(tmin, tmax + 1)
            g = rng.bounded(G)
            val = f['rows'][rng.bounded(N)][1] + rng.range(-100000, 100000)
            ins_t.append(t)
            ins_g.append(g)
            edits[v] = ('I', rid, t, g, val)
        elif roll < 90:
            rid = rng.range(0, N)
            edits[v] = ('D', rid)
        else:
            rid = rng.range(0, N)
            t = rng.range(tmin, tmax + 1)
            g = rng.bounded(G)
            edits[v] = ('M', rid, t, g, f['rows'][rid][1] + rng.range(-100000, 100000))
    return dict(queries=queries, parent=parent, edits=edits, ins_t=ins_t, ins_g=ins_g,
                n_ins=next_ins, N=N, G=G)


def run(f, w, audit_states):
    N, G = w['N'], w['G']
    n_rows = N + w['n_ins']
    t = [0] * n_rows
    g = [0] * n_rows
    amt = [0] * n_rows
    ex = [0] * n_rows
    for i in range(N):
        t[i], amt[i], g[i] = f['rows'][i]
        ex[i] = 1
    for k in range(w['n_ins']):
        t[N + k] = w['ins_t'][k]
        g[N + k] = w['ins_g'][k]
    fac = list(f['factors'])
    q = len(w['queries'])
    queries = w['queries']
    group_fac_cur = list(f['factors'])  # initial group factors
    # initial answer + mass matrix
    ans = [0] * q
    mass = [[0] * q for _ in range(G)]
    for rid in range(n_rows):
        if not ex[rid]:
            continue
        tt = t[rid]
        for qi in range(q):
            lo, hi = queries[qi]
            if lo <= tt < hi:
                ans[qi] += amt[rid] * fac[g[rid]]
                mass[g[rid]][qi] += amt[rid]
    answers = {0: list(ans)}
    audit = {}
    kids = [[] for _ in range(KSTATES)]
    for v in range(1, KSTATES):
        kids[w['parent'][v]].append(v)
    saved = [None] * KSTATES
    stack = [(0, False)]
    while stack:
        v, exiting = stack.pop()
        if exiting:
            if v > 0:
                sv = saved[v]
                if sv[0] == 'F':
                    _, gg, oldf = sv
                    for qi in range(q):
                        ans[qi] -= (fac[gg] - oldf) * mass[gg][qi]
                    fac[gg] = oldf
                else:
                    rid, oldt, oldg, olda, olde = sv
                    # remove the post-edit contribution, restore the pre-edit one
                    if ex[rid]:
                        for qi in range(q):
                            lo, hi = queries[qi]
                            if lo <= t[rid] < hi:
                                ans[qi] -= amt[rid] * fac[g[rid]]
                                mass[g[rid]][qi] -= amt[rid]
                    t[rid], g[rid], amt[rid], ex[rid] = oldt, oldg, olda, olde
                    if olde:
                        for qi in range(q):
                            lo, hi = queries[qi]
                            if lo <= oldt < hi:
                                ans[qi] += olda * fac[oldg]
                                mass[oldg][qi] += olda
            continue
        if v > 0:
            e = w['edits'][v]
            if e[0] == 'F':
                saved[v] = ('F', e[1], fac[e[1]])
                gg, nf = e[1], e[2]
                d = nf - fac[gg]
                for qi in range(q):
                    ans[qi] += d * mass[gg][qi]
                fac[gg] = nf
            else:
                rid = e[1]
                saved[v] = (rid, t[rid], g[rid], amt[rid], ex[rid])
                # remove old contribution
                if ex[rid]:
                    for qi in range(q):
                        lo, hi = queries[qi]
                        if lo <= t[rid] < hi:
                            ans[qi] -= amt[rid] * fac[g[rid]]
                            mass[g[rid]][qi] -= amt[rid]
                if e[0] == 'R':
                    amt[rid], ex[rid] = e[2], 1
                elif e[0] == 'I':
                    t[rid], g[rid], amt[rid], ex[rid] = e[2], e[3], e[4], 1
                elif e[0] == 'D':
                    ex[rid] = 0
                else:  # move
                    t[rid], g[rid], amt[rid], ex[rid] = e[2], e[3], e[4], 1
                if ex[rid]:
                    for qi in range(q):
                        lo, hi = queries[qi]
                        if lo <= t[rid] < hi:
                            ans[qi] += amt[rid] * fac[g[rid]]
                            mass[g[rid]][qi] += amt[rid]
            answers[v] = list(ans)
            if v in audit_states:
                brute = []
                for (lo, hi) in queries:
                    s = 0
                    for rid in range(n_rows):
                        if ex[rid] and lo <= t[rid] < hi:
                            s += amt[rid] * fac[g[rid]]
                    brute.append(s)
                assert brute == answers[v], (v, brute, answers[v])
                audit[v] = brute
            stack.append((v, True))
            for c in reversed(kids[v]):
                stack.append((c, False))
        else:
            stack.append((v, True))
            for c in reversed(kids[v]):
                stack.append((c, False))
    return {v: answers[v] for v in range(1, KSTATES)}, audit


def main():
    fixture = sys.argv[1]
    topology = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 7
    fp = int(sys.argv[4]) if len(sys.argv) > 4 else 37
    q = int(sys.argv[5]) if len(sys.argv) > 5 else 2
    out = sys.argv[6] if len(sys.argv) > 6 else None
    f = load_fixture(fixture)
    print(f"fixture rows={f['n']} groups={f['groups']} span={f['tmax'] - f['tmin'] + 1}", file=sys.stderr)
    w = make_workload(f, topology, seed, fp, q)
    print(f"workload q={q} n_ins={w['n_ins']}", file=sys.stderr)
    answers, audit = run(f, w, {1, 2, 3, 5, 17, 128, 777, 1024, 129, 2048})
    if out:
        with open(out, 'wb') as fp_out:
            for v in range(1, KSTATES):
                for x in answers[v]:
                    fp_out.write(struct.pack('<q', x))
        print(f"wrote {out}: {sum(len(answers[v]) for v in answers)} answers; audits={len(audit)}", file=sys.stderr)
    else:
        print(answers[2048])


if __name__ == '__main__':
    main()
