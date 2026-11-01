#!/usr/bin/env python3
"""Run independent jobs on separate GPUs and report lifecycle and launch rates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--binary', type=Path, default=Path('build/cuda/bin/gpu_snapshot'))
    parser.add_argument('--devices', default='0,1,2,3')
    parser.add_argument('--threads', type=int, default=32)
    parser.add_argument('--oracle-sha', required=True, help='Complete rollback-Fenwick answer SHA-256')
    parser.add_argument('--output', type=Path, default=Path('out/parallel-jobs'))
    args = parser.parse_args()
    if len(args.oracle_sha) != 64:
        parser.error('--oracle-sha requires a complete SHA-256')
    devices = args.devices.split(',')
    out = args.output
    out.mkdir(parents=True, exist_ok=True)

    def start(name, device):
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=device, OMP_NUM_THREADS=str(args.threads),
                   BWH_RANGE_SCOPE='per_state', BWH_V7_BLOCKS='32')
        command = [str(args.binary.resolve()), '--fixture', str(args.fixture.resolve()),
                   '--base-id', args.fixture.stem, '--topology', '0', '--seed', '7',
                   '--factor-period', '37', '--q', '2', '--repeat', '0',
                   '--answers', str(out / (name+'.bin')), '--json', str(out / (name+'.json'))]
        with (out / (name+'.log')).open('w') as log:
            return subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)

    def finish(process, name):
        if process.wait() != 0:
            raise RuntimeError(f'{name} failed; see its log')
        record = json.loads((out / (name+'.json')).read_text())
        payload = (out / (name+'.bin')).read_bytes()
        sha = hashlib.sha256(payload).hexdigest()
        if sha != args.oracle_sha or sha != record['answers_sha256']:
            raise ValueError(f'{name}: complete answer vector differs from reference')
        if len(payload) != 8*record['n_answers']:
            raise ValueError(f'{name}: answer count differs from worker metadata')
        record['fixture'] = args.fixture.name
        (out / (name+'.json')).write_text(json.dumps(record, indent=2)+'\n')
        return record

    solo = finish(start('solo', devices[0]), 'solo')
    begin = time.perf_counter()
    processes = [start(f'job{i}', device) for i, device in enumerate(devices)]
    codes = [process.wait() for process in processes]
    wall = time.perf_counter()-begin
    if any(codes):
        raise RuntimeError('Concurrent worker failed; see per-job logs')
    records = [finish(process, f'job{i}') for i, process in enumerate(processes)]
    median_ms = statistics.median(record['full_ns']/1e6 for record in records)
    rate = len(devices)*1000/median_ms
    result = dict(jobs=len(devices), solo_ms=solo['full_ns']/1e6,
                  per_job_full_ms=[record['full_ns']/1e6 for record in records],
                  per_job_kernel_ms=[record.get('gpu',{}).get('kernel_ns',0)/1e6 for record in records],
                  lifecycle_jobs_per_s=rate, lifecycle_gain=rate*solo['full_ns']/1e9,
                  launch_wall_s=wall, launch_jobs_per_s=len(devices)/wall,
                  answers_sha256=args.oracle_sha, exact=True)
    (out / 'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
