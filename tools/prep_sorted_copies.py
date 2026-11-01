#!/usr/bin/env python3
"""Write timestamp-sorted and (group,timestamp)-sorted copies of a BWBASE02 fixture.

Dataset preparation only (not charged to any arm): the row identity used by the workload stays
the original fixture; these copies are read-only layouts used to construct immutable base
prefixes with streaming writes. Both copies keep the header, reorder the base rows, and append
the original factor and label sections unchanged.
"""
import argparse
import struct
from pathlib import Path

import numpy as np

HDR = struct.Struct('<8s q i i q q q q q')
ROW = np.dtype([('t', '<i8'), ('a', '<i8'), ('g', '<i4'), ('p', '<i4')])


def write_copy(data, rec, order, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'wb') as f:
        f.write(data[:HDR.size])
        f.write(rec[order].tobytes())
        f.write(data[HDR.size + rec.nbytes:])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fixture', required=True)
    ap.add_argument('--out-t', required=True, help='timestamp-sorted copy')
    ap.add_argument('--out-gt', required=True, help='group,timestamp-sorted copy')
    args = ap.parse_args()
    data = Path(args.fixture).read_bytes()
    magic, n, groups, unit, lines, missing, tmin, tmax, scale = HDR.unpack_from(data, 0)
    assert magic == b'BWBASE02', magic
    rec = np.frombuffer(data, dtype=ROW, count=int(n), offset=HDR.size)
    ord_t = np.argsort(rec['t'], kind='stable')
    ord_gt = np.lexsort((rec['t'], rec['g']), axis=0)[0] if False else np.lexsort((rec['t'], rec['g']))
    write_copy(data, rec, ord_t, args.out_t)
    write_copy(data, rec, ord_gt, args.out_gt)
    print(f'{args.fixture}: {n} rows; wrote {args.out_t} and {args.out_gt}')


if __name__ == '__main__':
    main()
