#!/usr/bin/env python3
"""Convert the NYC TLC yellow-taxi month (public parquet) into a BWBASE02 fixture.

Prospective held-out base for BiWorldDB, 2026-10-25. Declared relation:
    R(id, ts, grp, amount) = one admitted trip row
      ts     : tpep_pickup_datetime as exact signed-64 microseconds since 2024-01-01
               (naive civil; the source timestamps are local civil timestamps without an
               assigned timezone, matching the Intel base convention)
      grp    : PULocationID (dense label kept in the fixture group_labels array)
      amount : total_amount in exact integer cents (round-half-away-from-zero of the
               published decimal amount; the published values are cent-granular)
    S(grp, multiplier) = base 1000 thousandths, as in every other base.
Admission predicate (declared, SQL-compatible): pickup timestamp inside the month,
PULocationID non-NULL, total_amount non-NULL. No imputation, no filtering of zero or
negative totals, no outlier removal.
"""
import argparse
import json
import struct

import numpy as np
import pyarrow.parquet as pq

MAGIC = b'BWBASE02'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parquet', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--month-start', default='2024-01-01T00:00:00')
    ap.add_argument('--month-end', default='2024-02-01T00:00:00')
    args = ap.parse_args()

    table = pq.read_table(args.parquet, columns=['tpep_pickup_datetime', 'PULocationID',
                                                 'total_amount'])
    n_raw = table.num_rows
    ts = table.column('tpep_pickup_datetime').combine_chunks()
    grp = table.column('PULocationID').combine_chunks()
    amt = table.column('total_amount').combine_chunks()

    ts_us = ts.cast('timestamp[us]').cast('int64').to_numpy(zero_copy_only=False)
    grp_np = grp.to_numpy(zero_copy_only=False).astype(np.int64)
    amt_np = amt.to_numpy(zero_copy_only=False).astype(np.float64)

    start_epoch = np.datetime64(args.month_start, 'us').astype('int64')
    end_epoch = np.datetime64(args.month_end, 'us').astype('int64')
    t = ts_us - start_epoch

    valid = (t >= 0) & (t < (end_epoch - start_epoch))
    valid &= ~np.isnan(amt_np)
    valid &= grp_np >= 0
    admitted = np.nonzero(valid)[0]
    t = t[admitted]
    grp_np = grp_np[admitted]
    amt_np = amt_np[admitted]

    # exact cents = round-half-away-from-zero(total_amount * 100)
    cents = np.sign(amt_np) * np.floor(np.abs(amt_np) * 100.0 + 0.5)
    cents = cents.astype(np.int64)

    labels = sorted(set(int(x) for x in np.unique(grp_np)))
    label_to_dense = {lab: i for i, lab in enumerate(labels)}
    dense = np.array([label_to_dense[int(x)] for x in grp_np], dtype=np.int32)

    n = int(t.size)
    G = len(labels)
    header = struct.pack('<8s q i i q q q q q', MAGIC, n, G, 0, n_raw,
                         n_raw - n, int(t.min()), int(t.max()), 2)
    rec = np.empty(n, dtype=np.dtype([('t', '<i8'), ('a', '<i8'), ('g', '<i4'), ('p', '<i4')]))
    rec['t'] = t
    rec['a'] = cents
    rec['g'] = dense
    rec['p'] = 0
    factors = np.full(G, 1000, dtype=np.int64)
    labels_arr = np.array(labels, dtype=np.int32)
    with open(args.out, 'wb') as fh:
        fh.write(header)
        fh.write(rec.tobytes())
        fh.write(factors.tobytes())
        fh.write(labels_arr.tobytes())

    meta = {
        'source': 'NYC TLC yellow_tripdata_2024-01.parquet (https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)',
        'source_rows': int(n_raw),
        'base_rows': n,
        'excluded_rows': int(n_raw - n),
        'groups': G,
        'group_labels': labels,
        'timestamp_unit': 'signed-64 naive-civil microseconds since 2024-01-01',
        'amount_scale': 2,
        'amount_unit': 'cents (round-half-away-from-zero of published total_amount)',
        'base_multiplier_thousandths': 1000,
        't_min': int(t.min()),
        't_max': int(t.max()),
        'timestamp_span': int(t.max() - t.min() + 1),
        'declared_predicate': 'pickup in month, PULocationID non-NULL, total_amount non-NULL',
        'note': 'Prospective held-out base: method, predicate, binaries, seeds and Q cohorts were frozen before this fixture was materialized; only the public schema and value domains needed for exact integer conversion were inspected.',
    }
    with open(args.out + '.meta.json', 'w') as fh:
        json.dump(meta, fh, indent=1)
        fh.write('\n')
    print(json.dumps({k: meta[k] for k in ('base_rows', 'excluded_rows', 'groups',
                                           'timestamp_span', 't_min', 't_max')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
