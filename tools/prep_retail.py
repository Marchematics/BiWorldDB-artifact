#!/usr/bin/env python3
"""Convert UCI Online Retail II into a BWBASE02 fixture.

Third prospective base for BiWorldDB, declared in THIRD_BASE_RETAIL_20261025.md before this
file was run. Declared relation:
    R(id, ts, grp, amount) = one admitted invoice line
      ts     : InvoiceDate as signed-64 integer minutes since 2009-12-01 00:00:00
      grp    : Country label, dense index by sorted unique label (labels kept)
      amount : round-half-away-from-zero(Quantity * UnitPrice * 100) integer cents
    S(grp, multiplier) = base 1000 thousandths.
Admission: InvoiceDate, Quantity, UnitPrice, Country non-NULL; both workbook sheets. No
filtering of negative quantities, zero prices or outliers; no imputation.
"""
import argparse
import hashlib
import json
import struct
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

MAGIC = b'BWBASE02'
SHEETS = ['Year 2009-2010', 'Year 2010-2011']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--xlsx', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--epoch', default='2009-12-01T00:00:00')
    args = ap.parse_args()

    src = Path(args.xlsx)
    sha = hashlib.sha256(src.read_bytes()).hexdigest()

    frames = []
    for sheet in SHEETS:
        df = pd.read_excel(src, sheet_name=sheet)
        df = df.rename(columns={'Invoice': 'InvoiceNo', 'Price': 'UnitPrice',
                                'Customer ID': 'CustomerID'})
        frames.append(df)
        print(f'{sheet}: {len(df)} rows, columns {list(df.columns)[:8]}')
    df = pd.concat(frames, ignore_index=True)
    n_raw = len(df)
    for col in ['InvoiceNo', 'StockCode', 'Quantity', 'InvoiceDate', 'UnitPrice', 'Country']:
        df[col] = df[col].where(df[col].notna())
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    valid = (df['InvoiceDate'].notna() & df['Quantity'].notna() & df['UnitPrice'].notna()
             & df['Country'].notna() & df['InvoiceNo'].notna() & df['StockCode'].notna())
    admitted = df[valid].copy()
    n = len(admitted)

    epoch = pd.Timestamp(args.epoch)
    minutes = ((admitted['InvoiceDate'] - epoch).dt.total_seconds() // 60).astype(np.int64)
    q = admitted['Quantity'].astype(np.float64).to_numpy()
    p = admitted['UnitPrice'].astype(np.float64).to_numpy()
    prod = q * p
    cents = np.sign(prod) * np.floor(np.abs(prod) * 100.0 + 0.5)
    cents = cents.astype(np.int64)

    labels = sorted(admitted['Country'].astype(str).unique().tolist())
    label_to_dense = {lab: i for i, lab in enumerate(labels)}
    dense = admitted['Country'].astype(str).map(label_to_dense).to_numpy(dtype=np.int32)
    t = minutes.to_numpy(dtype=np.int64)

    G = len(labels)
    header = struct.pack('<8s q i i q q q q q', MAGIC, n, G, 0, n_raw,
                         n_raw - n, int(t.min()), int(t.max()), 2)
    rec = np.empty(n, dtype=np.dtype([('t', '<i8'), ('a', '<i8'), ('g', '<i4'), ('p', '<i4')]))
    rec['t'] = t
    rec['a'] = cents
    rec['g'] = dense
    rec['p'] = 0
    factors = np.full(G, 1000, dtype=np.int64)
    labels_arr = np.array(range(G), dtype=np.int32)  # labels kept in meta (strings)
    with open(args.out, 'wb') as fh:
        fh.write(header)
        fh.write(rec.tobytes())
        fh.write(factors.tobytes())
        fh.write(labels_arr.tobytes())

    meta = {
        'source': 'UCI Online Retail II (DOI 10.24432/C5CG6D)',
        'source_archive_sha256': sha,
        'source_sheets': SHEETS,
        'source_rows': int(n_raw),
        'base_rows': int(n),
        'excluded_rows': int(n_raw - n),
        'groups': G,
        'group_labels': labels,
        'timestamp_unit': 'signed-64 integer minutes since 2009-12-01 00:00:00',
        'amount_scale': 2,
        'amount_unit': 'cents (round-half-away-from-zero of Quantity*UnitPrice)',
        'base_multiplier_thousandths': 1000,
        't_min': int(t.min()),
        't_max': int(t.max()),
        'timestamp_span': int(t.max() - t.min() + 1),
        'declared_predicate': 'InvoiceDate, InvoiceNo, StockCode, Quantity, UnitPrice and Country non-NULL on both sheets; no other filtering',
        'note': 'Third prospective base: declared in THIRD_BASE_RETAIL_20261025.md before this conversion ran; method, predicate, binaries, seeds and Q cohorts were already frozen.',
    }
    with open(args.out + '.meta.json', 'w') as fh:
        json.dump(meta, fh, indent=1)
        fh.write('\n')
    print(json.dumps({k: meta[k] for k in ('base_rows', 'excluded_rows', 'groups',
                                           'timestamp_span', 't_min', 't_max')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
