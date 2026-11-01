#!/usr/bin/env python3
"""Verify the frozen mixed-query policy, complete answers and measured choices."""
from pathlib import Path
import hashlib
import json
import math
import statistics

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'results/general_router_final2_20261102'
SHA = '10c5e1f92470c6d7313539ba4ae4da2145fbcfbbf5ea9b2310dcd37dd78da9c0'


def load(path):
    return json.loads(path.read_text())


def verify():
    policy_bytes = (DATA / 'policy_corrected.json').read_bytes()
    policy = json.loads(policy_bytes)
    receipt = load(DATA / 'policy_freeze_receipt.json')
    summary = load(DATA / 'summary.json')
    assert hashlib.sha256(policy_bytes).hexdigest() == receipt['policy_sha256'] == SHA
    assert (ROOT / 'policies/mixed.json').read_bytes() == policy_bytes
    assert receipt['frozen_before_load_generation'] and receipt['frozen_utc'] < summary['created']
    assert summary['reps'] == 2 and len(summary['cases']) == 9
    plans = []
    for cid, cell in summary['cases'].items():
        reference = (DATA / 'raw' / f'{cid}_oracle_r0.bin').read_bytes()
        assert reference and hashlib.sha256(reference).hexdigest() == cell['oracle']['answers_sha256']
        measured = {}
        for arm, recorded in cell['runs'].items():
            times = []
            for rep in range(2):
                rec = load(DATA / 'raw' / f'{cid}_{arm}_r{rep}.json')
                answer = (DATA / 'raw' / f'{cid}_{arm}_r{rep}.bin').read_bytes()
                assert rec['full_ns'] == sum(rec['phases_ns'].values())
                assert len(answer) == 8*rec.get('n_answers', rec.get('answers_n'))
                if 'answers_sha256' in rec:
                    assert hashlib.sha256(answer).hexdigest() == rec['answers_sha256']
                if arm not in cell['ineligible']:
                    assert answer == reference, (cid, arm, rep)
                else:
                    assert arm == 'union'
                    assert len(answer) == len(reference) == 32768
                    assert sum(answer[j:j+16] != reference[j:j+16]
                               for j in range(0, len(answer), 16)) == 2048
                times.append(rec['full_ns']/1e6)
            median = statistics.median(times)
            assert math.isclose(median, recorded['full_ms'], abs_tol=.001)
            if arm not in cell['ineligible']:
                measured[arm] = median
        chosen = cell['route']['chosen']
        assert measured[chosen] == min(measured.values())
        c = cell['case']
        if cell['class'] == '2d_rectangle':
            gate = policy['classes'][cell['class']]
            need = 8*c['groups']*(c['c0']+1)*(c['c1']+1)
            selected = ('densefen' if need <= gate['densefen_max_mb']*10**6 else
                        'fmd' if need <= gate['budget_mb']*10**6 and
                        c['c0']*c['c1'] <= gate['fmd_max_product'] else 'sparse2d')
            assert chosen == selected
        elif cell['class'] == 'adhoc_1d':
            assert chosen == policy['classes']['adhoc_1d']['default'] == 'deri'
            assert 'union' in policy['classes']['adhoc_1d']['ineligible']
        plans.append(cell['route']['plan_ns']/1000)
    regression = summary['frozen_v3_regression']
    assert regression['cases'] == 34 and regression['route_accuracy_vs_oracle'] == 1
    assert regression['regret_p95'] == 1.0448
    p95 = sorted(plans)[math.ceil(.95*(len(plans)-1))]
    print(f'PASS: nine exact choices; selected/best 1.0000 throughout; '
          f'planning median/P95 {statistics.median(plans):.2f}/{p95:.1f} us')


if __name__ == '__main__':
    verify()
