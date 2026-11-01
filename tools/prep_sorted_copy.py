#!/usr/bin/env python3
"""Write a time-ordered copy of a BWBASE02 fixture (dataset preparation, not charged).

The conversion keeps the header, sorts the base rows by timestamp and appends the original
factor and group-label sections unchanged. Row identity is preserved by the engine (edits still
address the original fixture); the sorted copy is only used to construct the class-major
prefix table in monotone class order.
"""
import argparse
import struct
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fixture', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    data = Path(args.fixture).read_bytes()
    magic, n, groups, unit, lines, missing, tmin, tmax, scale = struct.unpack_from(
        '<8s q i i q q q q q', data, 0)
    assert magic == b'BWBASE02', magic
    rec = np.frombuffer(data, dtype=np.dtype([('t', '<i8'), ('a', '<i8'),
                                              ('g', '<i4'), ('p', '<i4')]),
                        count=int(n), offset=64)
    order = np.argsort(rec['t'], kind='stable')
    out = bytearray()
    out += data[:64]
    out += rec[order].tobytes()
    out += data[64 + 24 * int(n):]
    Path(args.out).write_bytes(bytes(out))
    print(f'wrote {args.out}: {n} rows, sorted by timestamp')


if __name__ == '__main__':
    main()
