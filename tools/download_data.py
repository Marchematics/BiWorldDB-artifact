#!/usr/bin/env python3
"""Download a public dataset and convert it to the benchmark fixture format."""
import argparse
import gzip
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproduce import build

URLS = {
    'intel': 'https://db.csail.mit.edu/labdata/data.txt.gz',
    'taxi': 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet',
    'air': 'https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip',
    'retail': 'https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip',
}


def extract(archive, directory):
    with zipfile.ZipFile(archive) as z:
        for entry in z.infolist():
            path = (directory / entry.filename).resolve()
            if not path.is_relative_to(directory.resolve()):
                raise ValueError('Archive member outside dataset directory')
        z.extractall(directory)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('dataset', choices=URLS)
    args = ap.parse_args()
    data = ROOT / 'data' / args.dataset
    data.mkdir(parents=True, exist_ok=True)
    (ROOT / 'fixtures').mkdir(exist_ok=True)
    url = URLS[args.dataset]
    source = data / url.rsplit('/', 1)[-1]
    if not source.exists():
        print(f'Downloading {args.dataset}', flush=True)
        part = source.with_suffix(source.suffix + '.part')
        with urllib.request.urlopen(url, timeout=120) as response, part.open('wb') as output:
            shutil.copyfileobj(response, output)
        part.replace(source)
    names = {'intel': 'intel', 'taxi': 'taxi2024', 'air': 'uci501', 'retail': 'retail2009'}
    output = ROOT / 'fixtures' / (names[args.dataset] + '.bin')
    if args.dataset in ('intel', 'air'):
        bins = build(2049, ['prep'], 2)
        if args.dataset == 'intel':
            with gzip.open(source) as inp:
                with subprocess.Popen([bins / 'prep', '--format', 'intel', '--out', output],
                                      stdin=subprocess.PIPE) as process:
                    shutil.copyfileobj(inp, process.stdin)
                    process.stdin.close()
                    if process.wait(): raise RuntimeError('Intel conversion failed')
        else:
            extract(source, data)
            for nested in data.rglob('*.zip'):
                if nested != source: extract(nested, data)
            files = sorted(data.rglob('PRSA_Data_*.csv'))
            if not files: raise ValueError('Air-quality CSV files missing')
            subprocess.run([bins / 'prep', '--format', 'uci501', '--out', output, *files], check=True)
    elif args.dataset == 'taxi':
        subprocess.run([sys.executable, ROOT / 'tools/prep_taxi.py', '--parquet', source, '--out', output], check=True)
    else:
        extract(source, data)
        source = next(data.rglob('*.xlsx'))
        subprocess.run([sys.executable, ROOT / 'tools/prep_retail.py', '--xlsx', source, '--out', output], check=True)
    print(f'Fixture: fixtures/{output.name}')


if __name__ == '__main__':
    main()
