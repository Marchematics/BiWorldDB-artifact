#!/usr/bin/env python3
"""All-distinct joint planner v3 for the fast-preprocessing build (2026-11-01, P0).

v3 adds two calibration branches found by the v2-fast fresh miss: a factor-edit-count gate on
the shallow Q32 choice (few factor edits -> hybrid_fast beats aligned PGD-G's dense+corr path)
and an absolute-rows term at the Q128 rf/pgdg_align boundary.  Scores routing against the
*measured distinct-configuration oracle* for the current build.

Distinct configurations:
  hybrid       shipped hybrid worker, production env without BWH_CORR_PREFIX
  corr         shipped hybrid worker with BWH_CORR_PREFIX=auto (a distinct configuration only
               where the engine's corr-prefix gate is active: deep trees at Q<=2)
  batch_fast   the batch worker on a SHALLOW tree: its batch pass is depth-gated off, so it
               runs fast declared-request preprocessing + the shipped hybrid executor
               (exact, same answer vector; this is the new shallow work point)
  batch_deep   the batch worker on a DEEP tree (batch pass when the internal mode gate holds,
               otherwise its fast-preprocessing fallback)
  pgdg         PGD with the global weighted-deviation index
  rf           eager rollback Fenwick

The policy is frozen before any unseen fixture is generated.  Models: hybrid is fitted in
absolute log-ms; every other configuration as a log ratio against the hybrid on the same
cell.  `batch` measurements in the development file are re-labelled by state count.
"""
import argparse
import hashlib
import json
import math
import os
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import joint_planner as jp                                        # noqa: E402

POLICY_SCHEMA = "joint-planner-fast-v3"
DEEP_GATE_STATES = 8192
CONFIGS = ("hybrid_fast", "corr", "batch_deep", "pgdg_align", "rf")
# v2/v1 structural constants kept frozen
RF_DEEP_RPG_MAX = 4096.0
RF_SHALLOW_RPG_MAX = 1024.0
RF_SHALLOW_ROWS_MAX = 100000
RF_SHALLOW_FP_MAX = 4
RF_SHALLOW_FP_ROWS_MAX = 500000
PGDG_MARGIN = 0.99            # retained for audit; the fast-build rules below are structural
RF_SHALLOW_ROWS_SMALL = 8192  # q in [32,64): rf wins only on sub-8k-row bases
RF_SHALLOW_Q8_RPG = 850.0     # q==8: rf wins up to ~833 rows/group, pgdg_align beyond
RF_Q32_NF_MIN = 32            # v3: shallow 32<=Q<64 with n_factor <= this prefers hybrid_fast
RF_Q128_ROWS_MAX = 200000    # v3: Q128 rf only on <=200k-row bases (absolute-rows term)
# (no separate batch_fast work point: the standard hybrid path IS the fast-preprocessing path
#  now, and the batch worker's shallow route is the shipped hybrid executor)
DEEP_CONFIG = "batch_deep"
case_features = jp.case_features        # re-export for the bench wrapper
feature_vector = jp.vector


# ------------------------------------------------------------------ development relabelling
def relabel_measured(measured, states):
    """Relabel legacy `batch`/`pgdg` development arms when present."""
    out = dict(measured)
    if "batch" in out:
        out["batch_deep"] = out.pop("batch")
    if "pgdg" in out:
        out["pgdg_align"] = out.pop("pgdg")
    return out


def dev_cases(paths):
    cases = []
    for p in paths:
        doc = json.loads(Path(p).read_text())
        for c in doc["cases"]:
            c = dict(c)
            c["measured_ms"] = relabel_measured(c["measured_ms"], c["features"]["states"])
            cases.append(c)
    return cases


# ------------------------------------------------------------------ distinct candidate space
def corr_distinct(f):
    return int(f["states"]) >= DEEP_GATE_STATES and int(f["q"]) <= 2


def distinct_configs(f):
    states = int(f["states"])
    deep = states >= DEEP_GATE_STATES
    out = ["hybrid_fast", "pgdg_align", "rf"]
    if deep:
        out.append("batch_deep")
        if corr_distinct(f):
            out.append("corr")
    return out


def _rf_abs_ok(f, policy):
    rows = int(f["rows"])
    fp = int(f["factor_period"])
    return (rows <= int(policy.get("rf_shallow_rows_max", RF_SHALLOW_ROWS_MAX)) or
            (fp <= int(policy.get("rf_shallow_fp_max", RF_SHALLOW_FP_MAX)) and
             rows <= int(policy.get("rf_shallow_fp_rows_max", RF_SHALLOW_FP_ROWS_MAX))))


# ------------------------------------------------------------------ models
def fit_models(cases, ridge=1.0, base="hybrid_fast"):
    models = {}
    dat = [c for c in cases if c["measured_ms"].get(base)]
    models[base] = dict(jp._fit_one(dat, [math.log(float(c["measured_ms"][base])) for c in dat],
                                    ridge), kind="absolute", n_train=len(dat))

    def ratio(arm, region):
        rows = [c for c in cases if c["measured_ms"].get(arm) and c["measured_ms"].get(base)
                and region(c)]
        if len(rows) < 4:
            return None
        y = [math.log(float(c["measured_ms"][arm]) / float(c["measured_ms"][base])) for c in rows]
        return dict(jp._fit_one(rows, y, ridge), kind="ratio_vs_" + base, n_train=len(rows))

    deep = lambda c: int(c["features"]["states"]) >= DEEP_GATE_STATES
    shallow = lambda c: int(c["features"]["states"]) < DEEP_GATE_STATES
    models["corr"] = ratio("corr", lambda c: True)
    models["pgdg_align"] = ratio("pgdg_align", lambda c: True)
    models["rf"] = ratio("rf", lambda c: True)
    models["batch_deep"] = ratio("batch_deep", deep)
    return models


def predict(models, arm, f):
    m = models.get(arm)
    if not m:
        return None
    if m.get("kind") == "absolute":
        return jp._predict_abs(m, f)
    base = models.get("hybrid_fast")
    return jp._predict_abs(base, f) * jp._predict_abs(m, f)


# ------------------------------------------------------------------ decision
def _distinct_order(f, policy):
    """Structural rules for the fast-preprocessing build, defined on the development grid.

    deep (states >= 8192):
      q <= 2 -> batch_deep (batch v2 wins: 0.19-0.26 s vs pgdg_align 0.26-0.31 s and
                hybrid_fast 0.51-0.86 s)
      q >= 8 -> rf when rows/group <= rf_deep_rpg_max (measured rf 1.58-1.93 s vs hybrid
                3.76-5.52 s, batch fallback 4.26-5.98 s, pgdg_align 13-28 s); otherwise
                pgdg_align (low-group deep regime, outside the frozen unseen grid).
    shallow:
      q >= 64 -> rf when rows/group <= rf_shallow_rpg_max, else pgdg_align (dev Q128:
                 rf 54.55 vs pgdg 59.75 at 937 rows/group; pgdg 64.67 vs rf 76.21 at 1250)
      q in [32,64) -> rf only on bases with <= RF_SHALLOW_ROWS_SMALL rows (dev: tiny 2k
                rows rf 13.81 vs pgdg 26.10; 40k+ rows pgdg_align wins), else pgdg_align
      q == 8  -> rf when rows/group <= RF_SHALLOW_Q8_RPG (dev: rf wins at 125-833
                rows/group), else pgdg_align
      q <= 4  -> pgdg_align (dev: pgdg_align wins every q2/q4 cell)
    """
    states, q = int(f["states"]), int(f["q"])
    rows, groups = int(f["rows"]), int(f["groups"])
    rpg = rows / float(groups)
    if states >= int(policy.get("deep_gate_states", DEEP_GATE_STATES)):
        if q <= 2:
            return ["batch_deep", "pgdg_align", "hybrid_fast", "corr"], "deep_q<=2"
        if rpg <= float(policy.get("rf_deep_rpg_max", RF_DEEP_RPG_MAX)):
            return ["rf", "hybrid_fast", "batch_deep", "pgdg_align"], "deep_q>=8_rf_eligible"
        return ["pgdg_align", "hybrid_fast", "batch_deep", "rf"], "deep_q>=8_low_group"
    n_factor = (states - 1) // max(int(f["factor_period"]), 1)
    if q >= 64:
        # v3 absolute-rows term: dev Q128 rf win was at 120k rows / 937 rows-per-group; the
        # fresh miss was the same rows/group at 240k rows, where pgdg_align wins.  rf is kept
        # only on small absolute bases.
        if rpg <= float(policy.get("rf_shallow_rpg_max", RF_SHALLOW_RPG_MAX)) and \
                rows <= int(policy.get("rf_q128_rows_max", RF_Q128_ROWS_MAX)):
            return ["rf", "pgdg_align", "hybrid_fast"], "shallow_q>=64_rf_eligible"
        return ["pgdg_align", "rf", "hybrid_fast"], "shallow_q>=64"
    if q >= 32:
        if rows <= int(policy.get("rf_shallow_rows_small", RF_SHALLOW_ROWS_SMALL)):
            return ["rf", "pgdg_align", "hybrid_fast"], "shallow_q32_rf_small"
        # v3 factor-edit-count gate: low-nf cells (e.g. fp97 -> 21 factor edits) are won by
        # fast-preprocessing hybrid_fast; factor-heavy cells stay with aligned PGD-G.
        if n_factor <= int(policy.get("rf_q32_nf_min", RF_Q32_NF_MIN)):
            return ["hybrid_fast", "pgdg_align", "rf"], "shallow_q32_low_nf_hybrid"
        return ["pgdg_align", "hybrid_fast", "rf"], "shallow_q32"
    if q >= 8:
        if rpg <= float(policy.get("rf_shallow_q8_rpg_max", RF_SHALLOW_Q8_RPG)):
            return ["rf", "pgdg_align", "hybrid_fast"], "shallow_q8_rf_small"
        return ["pgdg_align", "hybrid_fast", "rf"], "shallow_q8"
    return ["pgdg_align", "hybrid_fast", "corr", "rf"], "shallow_q<=4"


def decide(policy, f):
    """Frozen fast-build decision: structural regime rule + candidate availability."""
    models = policy["models"]
    states, q = int(f["states"]), int(f["q"])
    deep = states >= int(policy.get("deep_gate_states", DEEP_GATE_STATES))
    cands = [a for a in distinct_configs(f) if models.get(a)]
    preds = {a: predict(models, a, f) for a in cands}
    preds = {a: v for a, v in preds.items() if v is not None}
    order, regime = _distinct_order(f, policy)
    chosen = next((a for a in order if a in preds), None)
    if chosen is None:
        chosen = min(preds, key=preds.get)
    return {
        "chosen": chosen,
        "regime": regime,
        "shortlist": [a for a in order if a in preds],
        "designated": order[0],
        "model_argmin": min(preds, key=preds.get),
        "predicted_ms": {a: round(preds[a], 3) for a in preds},
        "allowed": sorted(preds),
        "deep_tree": deep,
        "features": f,
    }


# ------------------------------------------------------------------ evaluation vs oracle
def _pct(v, q):
    if not v:
        return None
    idx = int(math.ceil(q * (len(v) - 1)))
    return v[min(len(v) - 1, max(0, idx))]


def evaluate(policy, cases, boot=10000, seed=20261031, quiet=False):
    rows = []
    for c in cases:
        f = c.get("features") or jp.case_features(c)
        measured = relabel_measured(c["measured_ms"], int(f["states"]))
        # oracle: best measured DISTINCT configuration (duplicates such as corr==hybrid are
        # only counted where the corr-prefix gate is actually active)
        distinct = [a for a in distinct_configs(f) if a in measured]
        if not distinct:
            distinct = list(measured)
        oracle = min(distinct, key=measured.get)
        dec = decide(policy, f)
        planned = c.get("planned_ms")
        if planned is None:
            planned = measured.get(dec["chosen"])
        routing = measured[dec["chosen"]] / measured[oracle] if measured[oracle] > 0 else None
        regret = planned / measured[oracle] if planned and measured[oracle] > 0 else None
        rows.append({
            "case": c.get("case_id") or c.get("base"), "base": f["base"],
            "topology": f["topology"], "seed": f["seed"], "factor_period": f["factor_period"],
            "q": f["q"], "states": f["states"], "rows": f["rows"], "groups": f["groups"],
            "measured_ms": {a: round(v, 3) for a, v in measured.items()},
            "distinct": distinct, "oracle": oracle, "oracle_ms": round(measured[oracle], 3),
            "chosen": dec["chosen"], "regime": dec["regime"],
            "predicted_ms": dec["predicted_ms"],
            "planned_ms": round(planned, 3) if planned else None,
            "chosen_forced_ms": round(measured.get(dec["chosen"], 0), 3),
            "route_correct": dec["chosen"] == oracle,
            "regret": round(regret, 4) if regret is not None else None,
            "regret_chosen_forced": round(routing, 4) if routing is not None else None,
            "plan_ns": c.get("plan_ns"), "rss_kb": c.get("rss_kb"),
        })
    reg = sorted(r["regret"] for r in rows if r["regret"] is not None)
    rout = sorted(r["regret_chosen_forced"] for r in rows if r["regret_chosen_forced"] is not None)
    n = len(rows)
    sel = {a: sum(1 for r in rows if r["chosen"] == a) for a in CONFIGS}
    summary = {
        "cases": n,
        "route_accuracy_vs_oracle": round(sum(r["route_correct"] for r in rows) / n, 4) if n else None,
        "regret_median": round(_pct(reg, 0.50), 4) if reg else None,
        "regret_p95": round(_pct(reg, 0.95), 4) if reg else None,
        "regret_max": round(reg[-1], 4) if reg else None,
        "regret_gt_1_05": sum(1 for v in reg if v > 1.05),
        "regret_gt_1_10": sum(1 for v in reg if v > 1.10),
        "routing_only_median": round(_pct(rout, 0.50), 4) if rout else None,
        "routing_only_p95": round(_pct(rout, 0.95), 4) if rout else None,
        "routing_only_max": round(rout[-1], 4) if rout else None,
        "selection_counts": sel,
        "p95_target_met": bool(reg and _pct(reg, 0.95) <= 1.05),
        "oracle": "best measured distinct configuration among {hybrid_fast,corr,batch_deep,pgdg_align,rf} "
                  "with corr counted only where its gate is active",
        "percentile_method": "nearest-rank sorted[ceil(q*(n-1))]; bootstrap over cases",
    }
    if boot and n:
        state = seed
        def rnd():
            nonlocal state
            state = (1103515245 * state + 12345) & 0x7FFFFFFF
            return state / float(0x7FFFFFFF)
        meds, p95s = [], []
        for _ in range(int(boot)):
            samp = sorted(reg[int(rnd() * n)] for _ in range(n))
            meds.append(_pct(samp, 0.50))
            p95s.append(_pct(samp, 0.95))
        meds.sort(); p95s.sort()
        lo = int(math.floor(0.025 * (len(meds) - 1)))
        hi = int(math.ceil(0.975 * (len(meds) - 1)))
        summary["regret_median_ci95"] = [round(meds[lo], 4), round(meds[hi], 4)]
        summary["regret_p95_ci95"] = [round(p95s[lo], 4), round(p95s[hi], 4)]
        summary["bootstrap_resamples"] = int(boot)
    out = {"policy_schema": policy.get("schema"), "summary": summary, "cases": rows}
    if not quiet:
        print(json.dumps(summary, indent=1))
    return out


# ------------------------------------------------------------------ freeze / CLI
def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fit(policy_meta, cases, ridge=1.0):
    models = fit_models(cases, ridge)
    pol = {
        "schema": POLICY_SCHEMA,
        "frozen": True,
        "created_utc": policy_meta.get("created_utc"),
        "frozen_by": "ext/joint_planner_fast_v3.py fit",
        "configs": list(CONFIGS),
        "deep_gate_states": DEEP_GATE_STATES,
        "feature_names": list(jp.FEATURE_NAMES),
        "ridge": ridge,
        "rf_deep_rpg_max": RF_DEEP_RPG_MAX,
        "rf_shallow_rpg_max": RF_SHALLOW_RPG_MAX,
        "rf_shallow_rows_max": RF_SHALLOW_ROWS_MAX,
        "rf_shallow_fp_max": RF_SHALLOW_FP_MAX,
        "rf_shallow_fp_rows_max": RF_SHALLOW_FP_ROWS_MAX,
        "pgdg_margin": PGDG_MARGIN,
        "rf_shallow_rows_small": RF_SHALLOW_ROWS_SMALL,
        "rf_shallow_q8_rpg_max": RF_SHALLOW_Q8_RPG,
        "rf_q32_nf_min": RF_Q32_NF_MIN,
        "rf_q128_rows_max": RF_Q128_ROWS_MAX,
        "regime_rules": {
            "deep_q<=2": "batch_deep first (batch pass applies; measured winner)",
            "deep_q>=8_rf_eligible": "rf first (rows/group <= rf_deep_rpg_max)",
            "deep_q>=8_low_group": "pgdg first",
            "shallow_q>=64_rf_eligible": "rf first when rows/group <= rf_shallow_rpg_max and rows <= rf_q128_rows_max",
            "shallow_q>=64": "pgdg_align first",
            "shallow_q32_rf_small": "rf first when rows <= rf_shallow_rows_small",
            "shallow_q32_low_nf_hybrid": "hybrid_fast first when n_factor <= rf_q32_nf_min",
            "shallow_q32": "pgdg_align first",
            "shallow_q8_rf_small": "rf first when rows/group <= rf_shallow_q8_rpg_max",
            "shallow_q8": "pgdg_align first",
            "shallow_q<=4": "pgdg_align first",
        },
        "models": models,
        "dev": {
            "cases": len(cases),
            "sources": policy_meta.get("sources", []),
            "measured_rows_per_config": {a: (models[a]["n_train"] if models[a] else 0)
                                         for a in CONFIGS},
        },
    }
    ev = evaluate(pol, cases, boot=2000, quiet=True)
    pol["dev"].update(ev["summary"])
    return pol


def freeze_receipt(policy_path, dev_cases_path, dev_runs):
    import time
    pol = json.loads(Path(policy_path).read_text())
    bench = str(HERE / "bench_joint_fast_v3.py")
    return {
        "frozen": True,
        "policy_path": str(policy_path),
        "policy_sha256": _sha256(policy_path),
        "policy_schema": pol.get("schema"),
        "policy_created_utc": pol.get("created_utc"),
        "freeze_recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code_sha256": {"joint_planner_fast_v3.py": _sha256(__file__),
                        "bench_joint_fast_v3.py": _sha256(bench) if os.path.exists(bench) else None},
        "dev_cases_path": str(dev_cases_path),
        "dev_cases_sha256": _sha256(dev_cases_path),
        "dev_runs": [{"path": str(p), "sha256": _sha256(p)} for p in dev_runs if os.path.exists(p)],
        "dev_summary": pol.get("dev"),
        "models": "hybrid absolute log-ms; corr/batch_fast/batch_deep/pgdg/rf as log ratios vs hybrid",
        "note": "The v3 unseen fixtures were NOT generated before this receipt. "
                "The run driver enforces the policy sha via --policy-sha.",
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="all-distinct joint planner")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--dev", nargs="+", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--ridge", type=float, default=1.0)
    f.add_argument("--created-utc", default="")
    z = sub.add_parser("freeze")
    z.add_argument("--policy", required=True)
    z.add_argument("--dev-cases", required=True)
    z.add_argument("--dev-runs", nargs="*", default=[])
    z.add_argument("--out", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--policy", required=True)
    e.add_argument("--cases", nargs="+", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--boot", type=int, default=10000)
    r = sub.add_parser("route")
    r.add_argument("--policy", required=True)
    r.add_argument("--case", required=True)
    c = sub.add_parser("check")
    c.add_argument("--policy", required=True)
    c.add_argument("--sha256", default="")
    a = ap.parse_args(argv)
    if a.cmd == "fit":
        cases = dev_cases(a.dev)
        created = a.created_utc or __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ",
                                                               __import__("time").gmtime())
        pol = fit({"created_utc": created, "sources": list(a.dev)}, cases, ridge=a.ridge)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(pol, indent=1) + "\n")
        print(json.dumps({"policy": a.out, "sha256": _sha256(a.out),
                          "dev_summary": pol["dev"]}, indent=1))
        return 0
    if a.cmd == "freeze":
        rec = freeze_receipt(a.policy, a.dev_cases, a.dev_runs)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rec, indent=1) + "\n")
        print(json.dumps({"receipt": a.out, "policy_sha256": rec["policy_sha256"],
                          "freeze_recorded_utc": rec["freeze_recorded_utc"]}, indent=1))
        return 0
    if a.cmd == "eval":
        pol = json.loads(Path(a.policy).read_text())
        cases = dev_cases(a.cases)
        out = evaluate(pol, cases, boot=a.boot)
        out["policy_sha256"] = _sha256(a.policy)
        Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
        print("wrote", a.out)
        return 0
    if a.cmd == "route":
        pol = json.loads(Path(a.policy).read_text())
        case = json.loads(Path(a.case).read_text())
        print(json.dumps(decide(pol, jp.case_features(case)), indent=1))
        return 0
    if a.cmd == "check":
        pol = json.loads(Path(a.policy).read_text())
        sha = _sha256(a.policy)
        ok = pol.get("schema") == POLICY_SCHEMA and pol.get("frozen") is True
        if a.sha256 and a.sha256 != sha:
            print(json.dumps({"ok": False, "expected": a.sha256, "actual": sha}))
            return 1
        print(json.dumps({"ok": ok, "sha256": sha, "created_utc": pol.get("created_utc"),
                          "dev": pol.get("dev", {}).get("cases")}))
        return 0 if ok else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
