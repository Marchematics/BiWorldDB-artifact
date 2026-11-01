#!/usr/bin/env python3
"""Prepare the retail SQL comparison and normalize client answer output."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import struct

import test_reference as ref


def prepare(args):
    fixture = ref.load_fixture(args.fixture)
    workload = ref.make_workload(fixture, 0, 7, 37, 2)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'r.csv').open('w', newline='') as stream:
        writer = csv.writer(stream, lineterminator="\n")
        for i, (t, amount, group) in enumerate(fixture['rows']):
            writer.writerow([i, t, group, amount, 1])
        for i in range(workload['n_ins']):
            writer.writerow([fixture['n'] + i, 0, 0, 0, 0])
    with (out / 's.csv').open('w', newline='') as stream:
        csv.writer(stream, lineterminator='\n').writerows(enumerate(fixture['factors']))
    schema = ('CREATE TABLE r(id BIGINT PRIMARY KEY, t BIGINT, g INT, amount BIGINT, present SMALLINT);\n'
              'CREATE TABLE s(g INT PRIMARY KEY, mul BIGINT);\n')
    (out / 'schema.sql').write_text(schema)
    (out / 'postgres_load.sql').write_text(
        "\\copy r FROM 'r.csv' WITH (FORMAT csv)\n\\copy s FROM 's.csv' WITH (FORMAT csv)\nANALYZE;\n")
    (out / 'mariadb_load.sql').write_text(''.join(
        f"LOAD DATA LOCAL INFILE '{table}.csv' INTO TABLE {table} FIELDS TERMINATED BY ',';\n"
        for table in ['r', 's']))
    # MonetDB COPY reads server-visible paths; colocated runs use the prepared directory.
    (out / 'monetdb_load.sql').write_text(''.join(
        f"COPY INTO {table} FROM '{str((out / (table+'.csv')).resolve()).replace(chr(39), chr(39)*2)}' "
        "USING DELIMITERS ',', '\\n';\n" for table in ['r', 's']))
    statements = []
    for v in range(1, ref.KSTATES):
        edit = workload['edits'][v]
        kind, target, *values = edit
        if kind == 'F':
            update = f'UPDATE s SET mul={values[0]} WHERE g={target};'
        elif kind == 'R':
            update = f'UPDATE r SET amount={values[0]},present=1 WHERE id={target};'
        elif kind == 'D':
            update = f'UPDATE r SET present=0 WHERE id={target};'
        else:
            t, group, amount = values
            update = f'UPDATE r SET t={t},g={group},amount={amount},present=1 WHERE id={target};'
        rng = ref.Rng((7*1000000007 + 37*131 + v*2654435761 + 17) & ref.MASK)
        ranges = ref.gen_nested(rng, fixture['tmin'], fixture['tmax'])[:2]
        columns = ','.join(f'COALESCE(SUM(CASE WHEN r.t>={lo} AND r.t<{hi} '
                           'THEN r.amount*s.mul END),0)' for lo, hi in ranges)
        statements.extend([update, f'SELECT {columns} FROM r JOIN s ON r.g=s.g WHERE r.present=1;'])
    (out / 'execute.sql').write_text('\n'.join(statements)+'\n')
    (out / 'metadata.json').write_text(json.dumps(dict(
        fixture=args.fixture.name, states=ref.KSTATES, topology='chain', seed=7,
        factor_period=37, q=2, n_answers=2*(ref.KSTATES-1)), indent=2)+'\n')


def normalize(args):
    pairs = re.findall(r'^\s*(-?\d+)\s*[|,\t]\s*(-?\d+)\s*$',
                       args.normalize.read_text(), re.M)
    if len(pairs) != ref.KSTATES-1:
        raise ValueError(f'Expected {ref.KSTATES-1} two-range rows, found {len(pairs)}')
    answers = b''.join(struct.pack('<q', int(value)) for pair in pairs for value in pair)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'answers.bin').write_bytes(answers)
    record = dict(n_answers=len(answers)//8, answers_sha256=hashlib.sha256(answers).hexdigest())
    if args.oracle:
        record['exact'] = answers == args.oracle.read_bytes()
        if not record['exact']:
            raise ValueError('SQL answers differ from the complete oracle vector')
    (args.output / 'answers.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--fixture', type=Path)
    source.add_argument('--normalize', type=Path, help='psql -At, mariadb -BN or mclient -fcsv output')
    parser.add_argument('--oracle', type=Path)
    parser.add_argument('--output', type=Path, default=Path('out/full-state-sql'))
    arguments = parser.parse_args()
    prepare(arguments) if arguments.fixture else normalize(arguments)
