#!/usr/bin/env python3
"""Frozen joint planner over five BiWorldDB execution work points (2026-10-31).

Work points (all keep the same offline declared-workload contract and answer vector):
  * hybrid : the shipped hybrid arm (panel.cpp --arm hybrid) with the production
             optimization environment but *without* BWH_CORR_PREFIX ("standard hybrid").
  * corr   : the same binary with BWH_CORR_PREFIX=auto ("correction placement / corr-prefix").
  * batch  : the compiled version-tree batch worker (panel_batch.cpp).  The worker itself
             gates the batch pass to kStates >= 8192 and falls back to the shipped hybrid
             with the fast batch preprocessing below that; the planner applies the same
             depth gate, so on shallow trees this candidate is not considered at all.
  * pgdg   : PGD with the global weighted-deviation index (arms_strong.hpp, panel_strong.cpp
             --arm pgdg).
  * rf     : eager rollback Fenwick (arms_classic.hpp, panel_classic.cpp --arm rf).

The planner consumes only cheap declared-workload features (a 64-byte fixture header plus the
declared workload knobs: states, topology, seed, factor period, q).  It never builds the
workload to plan.  A per-arm log-linear cost model is fitted on same-session development
measurements; structural candidate gates encode physical priors established by committed
evidence (batch depth gate, rf depth exclusion, tiny-base rf regime).  The fitted model, gates,
guards and tie-break order are serialized as a frozen policy file; the unseen-test driver loads
that exact file and records its sha256, so the policy cannot silently refit after the unseen
cases are known.

CLI
  fit    --dev dev_cases.json --out policy_frozen.json [--ridge 1.0]
  route  --policy policy_frozen.json --case case.json
  eval   --policy policy_frozen.json --cases scored_cases.json --out eval.json [--boot 10000]
  check  --policy policy_frozen.json [--sha256 SHA]     # freeze receipt enforcement
"""
import argparse
import hashlib
import json
import math
import os
import struct
import sys
from pathlib import Path

POLICY_SCHEMA = "joint-planner-policy-v1"
DEEP_GATE_STATES = 8192          # batch depth gate (same as batch_engine.hpp kStates >= 8192)
ARMS = ("hybrid", "corr", "batch", "pgdg", "rf")
# Feature basis: physical, no exact collinearity.  rows/groups/states/q/factor edits/row edits/
# span capture the algorithmic scaling; qv and rows give the fixed-cost terms.
FEATURE_NAMES = (
    "log_rows", "log_groups", "log_states", "log_q",
    "log_factor_edits", "log_row_edits", "log_span",
    "qv_over_1e5", "rows_over_1e6",
)
# Tie-break order among near-equal predicted candidates (physical preference, not fitted):
# deep trees prefer the batch pass; shallow trees prefer the shipped hybrid.
TIE_BREAK_DEEP = ("batch", "corr", "hybrid", "pgdg", "rf")
TIE_BREAK_SHALLOW = ("hybrid", "corr", "pgdg", "rf", "batch")
# Regime shortlists (defined on the development measurements + committed evidence).  The
# fitted cost models order the candidates *inside* the shortlist; the shortlist itself is the
# structural part of the policy and encodes the measured regime map.
RF_DEEP_RPG_MAX = 4096.0    # rows/group above which eager Fenwick reweighting dominates at depth
RF_SHALLOW_RPG_MAX = 1024.0  # shallow q>=32 crossing (dev f37: rf wins at 833, loses at 1250)
# v2 absolute-row gate for the shallow q>=32 rf decision (frozen v2 policy): the failure of
# the v1 rows/group-only rule at 200k rows/781 rows-per-group showed that rf's absolute
# construction cost must also be bounded.  rf is only considered when
#   rows <= rf_shallow_rows_max  OR  (factor_period <= rf_shallow_fp_max AND rows <= fp_rows_max)
RF_SHALLOW_ROWS_MAX = 100000
RF_SHALLOW_FP_MAX = 4
RF_SHALLOW_FP_ROWS_MAX = 500000
GUARDS = {}                 # shortlists replace the earlier margin guards

_HDR = struct.Struct("<8s q i i q q q q q")   # FixtureHeader, common.hpp (64 bytes)
_HDR_CACHE = {}


# ----------------------------------------------------------------------------- fixtures
def read_fixture_header(path):
    path = str(path)
    h = _HDR_CACHE.get(path)
    if h is None:
        with open(path, "rb") as f:
            raw = f.read(_HDR.size)
        if len(raw) != _HDR.size:
            raise ValueError("short fixture header: %s" % path)
        magic, n, groups, unit, orig, missing, t_min, t_max, amt_scale = _HDR.unpack(raw)
        if magic != b"BWBASE02":
            raise ValueError("bad fixture magic: %s" % path)
        h = {"rows": int(n), "groups": int(groups), "unit": int(unit),
             "t_min": int(t_min), "t_max": int(t_max), "amount_scale": int(amt_scale)}
        _HDR_CACHE[path] = h
    return h


def case_features(case, fixture_path=None):
    """Cheap declared-workload features.  No workload construction happens here."""
    fx = fixture_path or case.get("fixture")
    h = read_fixture_header(fx)
    states = int(case["states"])
    fp = int(case["factor_period"])
    q = int(case["q"])
    # edits occupy every state 1..states-1 (one per edited state); factor updates every fp-th
    # edit, and BWH_FACTOR_PCT is not used in this experiment, so the split is deterministic.
    factor_edits = (states - 1) // fp if fp > 0 else 0
    row_edits = (states - 1) - factor_edits
    span = h["t_max"] - h["t_min"] + 1
    return {
        "base": case.get("base", os.path.basename(fx)),
        "rows": h["rows"], "groups": h["groups"],
        "states": states, "q": q,
        "topology": int(case.get("topology", 0)),
        "seed": int(case.get("seed", 7)),
        "factor_period": fp,
        "factor_edits": factor_edits, "row_edits": row_edits,
        "span": span,
        "rows_per_group": h["rows"] / float(h["groups"]),
        "per_state": 1,
        "dense_predicate": 1 if (h["groups"] * span <= 2 * (h["rows"] + states - 1)) else 0,
    }


def vector(f):
    return [
        math.log(max(int(f["rows"]), 1)),
        math.log(max(int(f["groups"]), 1)),
        math.log(max(int(f["states"]), 1)),
        math.log(max(int(f["q"]), 1)),
        math.log(max(int(f["factor_edits"]), 1)),
        math.log(max(int(f["row_edits"]), 1)),
        math.log(max(int(f["span"]), 1)),
        int(f["states"]) * int(f["q"]) / 1e5,
        int(f["rows"]) / 1e6,
    ]


# ----------------------------------------------------------------------------- linear algebra
def _solve(A, b):
    """Small dense solve (Gaussian elimination with partial pivoting); n <= 11."""
    n = len(A)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        piv = M[c][c]
        if abs(piv) < 1e-12:
            continue
        inv = 1.0 / piv
        for r in range(n):
            if r == c:
                continue
            fac = M[r][c] * inv
            if fac:
                for k in range(c, n + 1):
                    M[r][k] -= fac * M[c][k]
    out = []
    for i in range(n):
        out.append(M[i][n] / M[i][i] if abs(M[i][i]) > 1e-12 else 0.0)
    return out


def _fit_one(dat, y, ridge):
    """Standardized ridge fit of a log-cost model; returns {w, mu, sd}."""
    X = [vector(c["features"]) for c in dat]
    n = len(X[0])
    mu = [sum(row[j] for row in X) / len(X) for j in range(n)]
    sd = []
    for j in range(n):
        var = sum((row[j] - mu[j]) ** 2 for row in X) / len(X)
        sd.append(math.sqrt(var) if var > 1e-24 else 1.0)
    Z = [[1.0] + [(row[j] - mu[j]) / sd[j] for j in range(n)] for row in X]
    A = [[0.0] * (n + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        for j in range(n + 1):
            A[i][j] = sum(Z[r][i] * Z[r][j] for r in range(len(Z)))
        if i == j and i > 0:
            A[i][j] += ridge
    bb = [sum(Z[r][i] * y[r] for r in range(len(Z))) for i in range(n + 1)]
    return {"w": _solve(A, bb), "mu": mu, "sd": sd}


def fit_models(cases, ridge=1.0, base_arm="hybrid"):
    """Per-work-point cost models.

    The shipped hybrid is modelled in absolute log-ms (it is measured on every cell);
    every other work point is modelled as a log *ratio* against the hybrid on the same
    cell.  The ratio basis cancels the shared setup terms (fixture scan, workload
    construction, answer extraction), which is exactly the part that accumulates
    session-to-session scale drift; it makes the routing decision robust to that drift.
    """
    models = {}
    # absolute model for the reference work point
    dat = [c for c in cases if c.get("measured_ms", {}).get(base_arm)]
    if len(dat) < 4:
        raise RuntimeError("not enough development rows for the %s reference model" % base_arm)
    models[base_arm] = dict(_fit_one(dat, [math.log(float(c["measured_ms"][base_arm]))
                                           for c in dat], ridge),
                            kind="absolute", n_train=len(dat))
    # ratio models for the other work points.  An arm is only fitted on the rows where
    # the structural gate makes it a candidate: the batch worker executed on a shallow
    # tree is its fallback route, not the batch pass, and must not bias the deep model.
    def in_candidate_region(arm, c):
        states = int(c["features"]["states"])
        if arm == "batch":
            return states >= DEEP_GATE_STATES
        return True

    for arm in ARMS:
        if arm == base_arm:
            continue
        dat = [c for c in cases
               if c.get("measured_ms", {}).get(arm) and c["measured_ms"].get(base_arm)
               and in_candidate_region(arm, c)]
        if len(dat) < 4:
            models[arm] = None
            continue
        y = [math.log(float(c["measured_ms"][arm]) / float(c["measured_ms"][base_arm]))
             for c in dat]
        models[arm] = dict(_fit_one(dat, y, ridge), kind="ratio_vs_" + base_arm,
                           n_train=len(dat))
    return models


def _predict_abs(m, f):
    v = vector(f)
    z = [1.0] + [(v[j] - m["mu"][j]) / m["sd"][j] for j in range(len(v))]
    return math.exp(sum(a * b for a, b in zip(m["w"], z)))


def predict(models, arm, f):
    m = models.get(arm)
    if not m:
        return None
    if m.get("kind") == "absolute":
        return _predict_abs(m, f)
    base = models.get("hybrid")
    if not base:
        return None
    return _predict_abs(base, f) * _predict_abs(m, f)


# ----------------------------------------------------------------------------- policy
def candidates(f, models):
    """Structural candidate set (physical gates, no fitted thresholds)."""
    states = int(f["states"])
    deep = states >= DEEP_GATE_STATES
    allow = {"hybrid": True, "corr": True, "pgdg": True}
    allow["batch"] = deep            # depth gate, same boundary as batch_engine.hpp
    # rf has no depth gate: its cost is dominated by construction plus the factor-edit
    # reweighting O(n_f * rows/group), not by depth alone (committed evidence shows rf
    # losing at 300k states on low-group bases but dev measurements show rf winning on
    # high-Q mid-depth trees).  The fitted ratio model decides, behind a 1.12x guard.
    allow["rf"] = True
    return {a: v for a, v in allow.items() if v and models.get(a)}


def shortlist(policy, f):
    """Structural regime shortlist (the frozen policy's measured regime map).

    deep trees (states >= deep_gate_states):
      * q <= 2  -> batch first: the batch pass applies exactly when the declared endpoint
        domain is small, which is the q<=2 regime; measured 2.7-4.7x over corr at 300k.
      * q >= 8  -> rf first when rows/group is small enough that rollback-Fenwick's
        factor-edit reweighting O(n_f * rows/group) stays cheap (measured rf wins at
        937-2344 rows/group, while the committed Intel/taxi deep cases at 11k-38k
        rows/group have rf at 11-21 s); otherwise PGD-G first (global deviation index
        removes the per-query dirty-group loop, its factor edits are O(nonzero deltas)).
    shallow trees (states < deep_gate_states):
      * q <= 8  -> the shipped hybrid (corr-prefix auto is inactive below 8192 states)
      * q >= 32 -> rf first when rows/group <= 1024 AND the v2 absolute-row gate holds
        (rows <= 100k, or factor_period <= 4 and rows <= 500k); otherwise the
        batch/hybrid shortlist (batch is gated out on shallow trees, so hybrid runs).
    """
    states, q = int(f["states"]), int(f["q"])
    rpg = f["rows"] / float(f["groups"])
    deep = states >= int(policy.get("deep_gate_states", DEEP_GATE_STATES))
    if deep:
        if q <= 2:
            return ("batch", "corr", "hybrid"), "deep_q<=2"
        if rpg <= float(policy.get("rf_deep_rpg_max", RF_DEEP_RPG_MAX)):
            return ("rf", "corr", "hybrid", "batch"), "deep_q>=8_rf_eligible"
        return ("pgdg", "corr", "hybrid", "batch"), "deep_q>=8_low_group"
    if q <= 8:
        return ("hybrid", "corr"), "shallow_q<=8"
    rows = int(f["rows"])
    fp = int(f["factor_period"])
    rf_small = (rpg <= float(policy.get("rf_shallow_rpg_max", RF_SHALLOW_RPG_MAX)))
    # v2 rule: rf's construction cost depends on absolute rows as well as rows/group.
    # From the dev grid: rf wins at <=40k rows for f37/f4, at 120k/320k for f4, but loses
    # at 120k+ for f37 and at 1.2M for f4.  The old rows/group-only rule picked rf at
    # 200k rows/781 rows-per-group where the shipped/batch worker is 1.2-1.3x faster.
    rf_abs_ok = (rows <= int(policy.get("rf_shallow_rows_max", RF_SHALLOW_ROWS_MAX)) or
                 (fp <= int(policy.get("rf_shallow_fp_max", RF_SHALLOW_FP_MAX)) and
                  rows <= int(policy.get("rf_shallow_fp_rows_max", RF_SHALLOW_FP_ROWS_MAX))))
    if rf_small and rf_abs_ok:
        return ("rf", "hybrid", "corr"), "shallow_q>=32_rf_eligible"
    return ("batch", "hybrid", "corr", "rf"), "shallow_q>=32_large"


def decide(policy, f):
    """Return the frozen-policy routing decision for one declared workload.

    The structural shortlist fixes the regime; the fitted log-cost models order the
    shortlisted work points and supply the predicted times recorded with the decision.
    """
    models = policy["models"]
    allowed = candidates(f, models)
    order, regime = shortlist(policy, f)
    preds = {a: predict(models, a, f) for a in allowed if models.get(a) is not None}
    preds = {a: v for a, v in preds.items() if v is not None}
    # The regime shortlist names the designated work point; the fitted cost model only
    # supplies the predicted-time forecast recorded with the decision.  If the designated
    # arm is not eligible/fitted (e.g. batch on a shallow tree where the depth gate removes
    # it) fall through to the next shortlisted arm, and only then to the model argmin.
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
        "deep_tree": int(f["states"]) >= int(policy.get("deep_gate_states", DEEP_GATE_STATES)),
        "features": f,
    }


def fit(policy_meta, dev_cases, ridge=1.0):
    models = fit_models(dev_cases, ridge)
    policy = {
        "schema": POLICY_SCHEMA,
        "frozen": True,
        "created_utc": policy_meta.get("created_utc"),
        "frozen_by": "ext/joint_planner.py fit",
        "work_points": list(ARMS),
        "deep_gate_states": DEEP_GATE_STATES,
        "feature_names": list(FEATURE_NAMES),
        "ridge": ridge,
        "guards": dict(GUARDS),
        "rf_deep_rpg_max": RF_DEEP_RPG_MAX,
        "rf_shallow_rpg_max": RF_SHALLOW_RPG_MAX,
        "rf_shallow_rows_max": RF_SHALLOW_ROWS_MAX,
        "rf_shallow_fp_max": RF_SHALLOW_FP_MAX,
        "rf_shallow_fp_rows_max": RF_SHALLOW_FP_ROWS_MAX,
        "regime_rules": {
            "deep_q<=2": "batch first (depth gate; q<=2 is the small-class-domain batch regime)",
            "deep_q>=8_rf_eligible": "rf first (rows/group <= rf_deep_rpg_max), corr/hybrid/batch behind",
            "deep_q>=8_low_group": "pgdg first (rows/group > rf_deep_rpg_max), corr/hybrid behind",
            "shallow_q<=8": "hybrid first (corr-prefix auto inactive below 8192 states)",
            "shallow_q>=32_rf_eligible": "rf first (rows/group <= rf_shallow_rpg_max AND "
                                          "the v2 absolute-row gate)",
            "shallow_q>=32_large": "batch/hybrid shortlist (batch gated out below 8192 states, "
                                   "so the shipped hybrid runs)",
        },
        "tie_break_deep": list(TIE_BREAK_DEEP),
        "tie_break_shallow": list(TIE_BREAK_SHALLOW),
        "models": models,
        "dev": {
            "cases": len(dev_cases),
            "sources": policy_meta.get("sources", []),
            "measured_rows_per_arm": {a: (models[a]["n_train"] if models[a] else 0) for a in ARMS},
        },
    }
    policy["dev"].update(evaluate(policy, dev_cases, boot=2000, quiet=True)["summary"])
    return policy


# ----------------------------------------------------------------------------- evaluation
def corr_distinct(f):
    """True when BWH_CORR_PREFIX=auto actually engages on this declared workload.

    The engine gate is kStates >= 8192 AND n_factor_edits*C <= 6e7 AND
    n_factor_edits*C <= 2*query_work, with C = declared endpoint domain.  In the measured
    grids the endpoint domain is <= ~4.3k classes only in the deep q<=2 regime; at q>=8 it
    is 3.1M-17.5M classes, so the gate is provably inactive there (n_f*C >> 6e7).
    """
    return int(f["states"]) >= 8192 and int(f["q"]) <= 2


def _percentile(sorted_vals, q):
    if not sorted_vals:
        return None
    idx = int(math.ceil(q * (len(sorted_vals) - 1)))
    return sorted_vals[min(len(sorted_vals) - 1, max(0, idx))]


def evaluate(policy, cases, boot=10000, seed=20261031, quiet=False):
    rows = []
    for c in cases:
        f = c.get("features") or case_features(c)
        dec = decide(policy, f)
        measured = c["measured_ms"]
        # Distinct execution configurations.  The correction-placement work point is a
        # distinct configuration only where the engine's corr-prefix auto gate is active:
        # kStates >= 8192 AND n_factor_edits*C <= 6e7 (C = declared endpoint domain).  C
        # grows combinatorially with q (4,324 at q2, 3.1M at q8, 17.5M at q32 for 300k
        # states), so in this frozen grid the gate is active only on deep q<=2 cells; on
        # shallow trees and deep q>=8 the corr worker executes the shipped hybrid path with
        # the env switch inactive.  Counting it again as an independent forced arm would
        # compare two samples of the same configuration and inject pure run-to-run noise
        # into the yardstick (observed: identical paths differ by up to 14% in low-ms cells).
        distinct = dict(measured)
        if not corr_distinct(f) and "corr" in distinct:
            distinct.pop("corr")
        best_all = min(distinct, key=distinct.get)
        # Gate-consistent yardstick: the planner is only allowed to route to arms that pass
        # the frozen structural gates, so the primary regret compares against the best
        # *eligible* forced arm.  The gated-out arms are still measured and reported as a
        # secondary yardstick (e.g. the batch worker on a shallow tree runs its shipped-hybrid
        # fallback, which the policy deliberately does not consider a batch work point).
        allowed = [a for a in candidates(f, policy["models"]) if a in measured and a in distinct]
        if not allowed:
            allowed = list(distinct)
        best = min(allowed, key=measured.get)
        planned = c.get("planned_ms")
        if planned is None:      # fall back to the chosen arm's measured median (dev in-sample)
            planned = measured[dec["chosen"]]
        regret = planned / measured[best] if measured[best] > 0 else None
        regret_all = planned / measured[best_all] if measured[best_all] > 0 else None
        rows.append({
            "case": c.get("case_id") or c.get("base"),
            "base": f["base"], "topology": f["topology"], "seed": f["seed"],
            "factor_period": f["factor_period"], "q": f["q"], "states": f["states"],
            "rows": f["rows"], "groups": f["groups"],
            "chosen": dec["chosen"], "eligible": allowed, "best_forced": best,
            "best_forced_all_arms": best_all,
            "route_correct": dec["chosen"] == best,
            "route_correct_all_arms": dec["chosen"] == best_all,
            "route_within_1pct": measured[dec["chosen"]] <= measured[best] * 1.01,
            "planned_ms": round(planned, 3),
            "best_forced_ms": round(measured[best], 3),
            "best_forced_all_arms_ms": round(measured[best_all], 3),
            "chosen_forced_ms": round(measured[dec["chosen"]], 3),
            "measured_ms": {a: round(v, 3) for a, v in measured.items()},
            "predicted_ms": dec["predicted_ms"],
            "regret": round(regret, 4) if regret is not None else None,
            "regret_all_arms": round(regret_all, 4) if regret_all is not None else None,
            "plan_ns": c.get("plan_ns"),
            "rss_kb": c.get("rss_kb"),
        })
    regret_vals = sorted(r["regret"] for r in rows if r["regret"] is not None)
    regret_all_vals = sorted(r["regret_all_arms"] for r in rows if r["regret_all_arms"] is not None)
    n = len(regret_vals)
    correct = sum(1 for r in rows if r["route_correct"])
    correct_all = sum(1 for r in rows if r["route_correct_all_arms"])
    within = sum(1 for r in rows if r["route_within_1pct"])
    sel = {a: sum(1 for r in rows if r["chosen"] == a) for a in ARMS}

    summary = {
        "cases": n,
        "route_accuracy_strict": round(correct / n, 4) if n else None,
        "route_accuracy_all_arms": round(correct_all / n, 4) if n else None,
        "route_accuracy_within_1pct": round(within / n, 4) if n else None,
        "regret_median": round(_percentile(regret_vals, 0.50), 4) if n else None,
        "regret_p95": round(_percentile(regret_vals, 0.95), 4) if n else None,
        "regret_max": round(regret_vals[-1], 4) if n else None,
        "regret_all_arms_median": round(_percentile(regret_all_vals, 0.50), 4) if n else None,
        "regret_all_arms_p95": round(_percentile(regret_all_vals, 0.95), 4) if n else None,
        "regret_all_arms_max": round(regret_all_vals[-1], 4) if n else None,
        "regret_gt_1_05": sum(1 for v in regret_vals if v > 1.05),
        "regret_gt_1_10": sum(1 for v in regret_vals if v > 1.10),
        "selection_counts": sel,
        "p95_target_met": bool(n and _percentile(regret_vals, 0.95) <= 1.05),
        "p95_target_met_all_arms": bool(n and _percentile(regret_all_vals, 0.95) <= 1.05),
        "regret_yardstick": "primary = best eligible forced arm (frozen structural gates); "
                            "all_arms = best forced arm over every measured work point",
        "percentile_method": "nearest-rank: sorted[ceil(q*(n-1))]; bootstrap over cases",
    }
    if boot and n:
        # bootstrap CI for median and P95 regret, resampling cases with replacement
        state = seed
        def rnd():
            nonlocal state
            state = (1103515245 * state + 12345) & 0x7FFFFFFF
            return state / float(0x7FFFFFFF)
        meds, p95s = [], []
        for _ in range(int(boot)):
            samp = [regret_vals[int(rnd() * n)] for _ in range(n)]
            samp.sort()
            meds.append(_percentile(samp, 0.50))
            p95s.append(_percentile(samp, 0.95))
        meds.sort(); p95s.sort()
        lo = int(math.floor(0.025 * (len(meds) - 1)))
        hi = int(math.ceil(0.975 * (len(meds) - 1)))
        summary["regret_median_ci95"] = [round(meds[lo], 4), round(meds[hi], 4)]
        summary["regret_p95_ci95"] = [round(p95s[lo], 4), round(p95s[hi], 4)]
        summary["bootstrap_resamples"] = int(boot)
        meds2, p95s2 = [], []
        for _ in range(int(boot)):
            samp = [regret_all_vals[int(rnd() * n)] for _ in range(n)]
            samp.sort()
            meds2.append(_percentile(samp, 0.50))
            p95s2.append(_percentile(samp, 0.95))
        meds2.sort(); p95s2.sort()
        summary["regret_all_arms_p95_ci95"] = [round(p95s2[lo], 4), round(p95s2[hi], 4)]
    out = {"policy_schema": policy.get("schema"), "summary": summary, "cases": rows}
    if not quiet:
        print(json.dumps(summary, indent=1))
    return out


# ----------------------------------------------------------------------------- dev merge / freeze
def merge_dev_runs(paths, created_utc=None):
    """Merge one or more ext/bench_joint.py run JSONs into a development case file.

    Repeated measurements of the same (case, arm) are reduced with the median; the case
    features are recomputed from the fixture header + declared workload knobs, so the cost
    model is fitted on exactly the features the planner sees at routing time.
    """
    import statistics
    merged = {}
    for path in paths:
        doc = json.loads(Path(path).read_text())
        for cid, entry in doc.get("cases", {}).items():
            spec = dict(entry["spec"])
            case = dict(spec)
            case["fixture"] = spec.get("fixture") or os.path.join(
                os.environ.get("JP_FIX", "/tmp/jp_jh/fix"), spec["base"] + ".bin")
            ftr = case_features(case)
            rec = merged.setdefault(cid, {"case_id": cid, "features": ftr, "samples": {},
                                          "sources": []})
            if path not in rec["sources"]:
                rec["sources"].append(path)
            for arm, runs in entry.get("arms", {}).items():
                for r in runs:
                    if "full_ms" in r:
                        rec["samples"].setdefault(arm, []).append(r["full_ms"])
    cases = []
    for cid in sorted(merged):
        rec = merged[cid]
        cases.append({
            "case_id": cid, "features": rec["features"],
            "measured_ms": {a: round(statistics.median(v), 3)
                            for a, v in sorted(rec["samples"].items())},
            "n_samples": {a: len(v) for a, v in sorted(rec["samples"].items())},
            "sources": rec["sources"],
        })
    return {"created_utc": created_utc, "sources": [str(p) for p in paths], "cases": cases}


def freeze_receipt(policy_path, dev_cases_path, dev_runs):
    import time
    pol = json.loads(Path(policy_path).read_text())
    sha = _sha256_file(policy_path)
    code = {os.path.basename(__file__): _sha256_file(__file__)}
    bench = os.path.join(os.path.dirname(__file__), "bench_joint.py")
    if os.path.exists(bench):
        code["bench_joint.py"] = _sha256_file(bench)
    dev_doc = json.loads(Path(dev_cases_path).read_text())
    return {
        "frozen": True,
        "policy_path": str(policy_path),
        "policy_sha256": sha,
        "policy_schema": pol.get("schema"),
        "policy_created_utc": pol.get("created_utc"),
        "freeze_recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code_sha256": code,
        "dev_cases_path": str(dev_cases_path),
        "dev_cases_sha256": _sha256_file(dev_cases_path),
        "dev_runs": [{"path": str(p), "sha256": _sha256_file(p),
                      "started_utc": json.loads(Path(p).read_text()).get("started_utc"),
                      "finished_utc": json.loads(Path(p).read_text()).get("finished_utc")}
                     for p in dev_runs if os.path.exists(p)],
        "dev_summary": pol.get("dev"),
        "measured_rows_per_arm": pol.get("dev", {}).get("measured_rows_per_arm"),
        "model_form": "log-linear ridge; hybrid modelled in absolute log-ms, every other work "
                      "point as a log ratio against the hybrid on the same cell",
        "dev_case_count": len(dev_doc.get("cases", [])),
        "note": "The unseen grid is NOT measured before this receipt: the unseen driver "
                "enforces this sha256 via --policy-sha and records it in its run meta.",
    }

# ----------------------------------------------------------------------------- CLI
def _load(path):
    return json.loads(Path(path).read_text())


def _dump(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1) + "\n")


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fit", help="fit the per-arm cost models and write the frozen policy")
    f.add_argument("--dev", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--ridge", type=float, default=1.0)
    f.add_argument("--sources", default="")

    r = sub.add_parser("route", help="route one declared workload with a frozen policy")
    r.add_argument("--policy", required=True)
    r.add_argument("--case", required=True)

    e = sub.add_parser("eval", help="score a frozen policy on cases with measured medians")
    e.add_argument("--policy", required=True)
    e.add_argument("--cases", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--boot", type=int, default=10000)

    c = sub.add_parser("check", help="verify a frozen policy file (schema + optional sha256)")
    c.add_argument("--policy", required=True)
    c.add_argument("--sha256", default="")

    m = sub.add_parser("merge", help="merge bench_joint run JSONs into a dev case file")
    m.add_argument("--runs", nargs="+", required=True)
    m.add_argument("--out", required=True)

    z = sub.add_parser("freeze", help="emit the freeze receipt for a fitted policy")
    z.add_argument("--policy", required=True)
    z.add_argument("--dev-cases", required=True)
    z.add_argument("--dev-runs", nargs="*", default=[])
    z.add_argument("--out", required=True)

    a = ap.parse_args(argv)
    if a.cmd == "fit":
        dev = _load(a.dev)
        dev_cases = dev["cases"] if isinstance(dev, dict) else dev
        policy = fit({"created_utc": dev.get("created_utc") if isinstance(dev, dict) else None,
                      "sources": [s for s in a.sources.split(",") if s]},
                     dev_cases, ridge=a.ridge)
        _dump(policy, a.out)
        sha = _sha256_file(a.out)
        print(json.dumps({"policy": a.out, "sha256": sha,
                          "bytes": os.path.getsize(a.out),
                          "summary": policy["dev"]}, indent=1))
        return 0
    if a.cmd == "route":
        policy = _load(a.policy)
        case = _load(a.case)
        ftr = case_features(case)
        dec = decide(policy, ftr)
        dec["policy_sha256"] = _sha256_file(a.policy)
        print(json.dumps(dec, indent=1))
        return 0
    if a.cmd == "eval":
        policy = _load(a.policy)
        cases = _load(a.cases)
        cases = cases["cases"] if isinstance(cases, dict) else cases
        out = evaluate(policy, cases, boot=a.boot)
        out["policy_sha256"] = _sha256_file(a.policy)
        out["policy_path"] = a.policy
        _dump(out, a.out)
        return 0
    if a.cmd == "check":
        policy = _load(a.policy)
        ok = policy.get("schema") == POLICY_SCHEMA and policy.get("frozen") is True
        sha = _sha256_file(a.policy)
        if a.sha256 and a.sha256 != sha:
            print(json.dumps({"ok": False, "reason": "sha256 mismatch",
                              "expected": a.sha256, "actual": sha}))
            return 1
        print(json.dumps({"ok": ok, "sha256": sha, "created_utc": policy.get("created_utc"),
                          "dev": policy.get("dev", {}).get("cases"),
                          "measured_rows_per_arm": policy.get("dev", {}).get("measured_rows_per_arm")}))
        return 0 if ok else 1
    if a.cmd == "merge":
        import time
        doc = merge_dev_runs(a.runs, created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                               time.gmtime()))
        _dump(doc, a.out)
        print(json.dumps({"out": a.out, "cases": len(doc["cases"]),
                          "arms": {arm: sum(1 for c in doc["cases"]
                                            if arm in c["measured_ms"]) for arm in ARMS}},
                         indent=1))
        return 0
    if a.cmd == "freeze":
        rec = freeze_receipt(a.policy, a.dev_cases, a.dev_runs)
        _dump(rec, a.out)
        print(json.dumps({"receipt": a.out, "policy_sha256": rec["policy_sha256"],
                          "freeze_recorded_utc": rec["freeze_recorded_utc"],
                          "dev_case_count": rec["dev_case_count"],
                          "dev_summary": rec["dev_summary"]}, indent=1))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
