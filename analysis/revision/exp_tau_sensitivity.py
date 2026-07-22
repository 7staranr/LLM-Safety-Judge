"""Reviewer 1, Q1: sensitivity of judge selection to the Stage-1 floor tau_min.

For tau_min in {0.40, 0.45, 0.50, 0.55, 0.60} and all seven prompt sources
(K = 9 judges), this reports:
  * per-source Stage-1 pass count and the per-cell status flips vs tau = 0.50;
  * the full-data R5 selection (CI filter + BA tie-break, abstain otherwise);
  * 5-fold cross-fitted held-out MAE of the R5 selection at each tau.

Run:  python exp_tau_sensitivity.py
"""
import json
import os
import sys

from sklearn.model_selection import KFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

from protocol.data import ENGLISH_SOURCES, JUDGES, SOURCES, calibrate, load_source
from protocol.judge_calibration import NMIN_UNSAFE

TAUS = [0.40, 0.45, 0.50, 0.55, 0.60]
ROOT = os.environ.get("DISTILL_ROOT", os.path.abspath(os.path.join(HERE, "..", "..")))
OUT = os.path.join(HERE, "results", "tau_sensitivity.json")


def stage1(m, tau):
    if m is None:
        return None
    if m.n_unsafe < NMIN_UNSAFE:
        return "UNDERPOWERED"
    return "PASS" if m.ci_lb >= tau else "FAIL"


def select_r5(metrics, tau):
    passed = {j: m for j, m in metrics.items() if stage1(m, tau) == "PASS"}
    if not passed:
        return None
    return sorted(passed, key=lambda j: (-passed[j].ba, j))[0]


def condition_error(group, judge):
    pairs = [(r["true_unsafe"], r["preds"].get(judge)) for r in group
             if r["true_unsafe"] is not None and r["preds"].get(judge) is not None]
    if not pairs:
        return None
    human = sum(1 for t, _ in pairs if t) / len(pairs)
    judged = sum(1 for _, p in pairs if p) / len(pairs)
    return abs(judged - human)


def fold_mae(records, judge):
    by_ratio = {}
    for r in records:
        by_ratio.setdefault(r["ratio"], []).append(r)
    errs = [e for e in (condition_error(g, judge) for g in by_ratio.values()) if e is not None]
    return sum(errs) / len(errs) if errs else None


def main():
    per_source_records = {s: load_source(s, ROOT) for s in SOURCES}
    full_metrics = {s: calibrate(recs) for s, recs in per_source_records.items()}

    out = {"taus": TAUS, "by_tau": {}}
    base_status = {}
    for s in SOURCES:
        for j in JUDGES:
            base_status[(s, j)] = stage1(full_metrics[s].get(j), 0.50)

    for tau in TAUS:
        info = {"per_source": {}, "flips_vs_050": []}
        for s in SOURCES:
            statuses = {j: stage1(full_metrics[s].get(j), tau) for j in JUDGES}
            n_pass = sum(1 for v in statuses.values() if v == "PASS")
            sel = select_r5(full_metrics[s], tau)
            info["per_source"][s] = {"n_pass": n_pass, "r5_selected": sel, "status": statuses}
            for j in JUDGES:
                a, b = base_status[(s, j)], statuses[j]
                if a != b and None not in (a, b):
                    info["flips_vs_050"].append(
                        {"source": s, "judge": j, "at_050": a, "at_tau": b,
                         "ci_lb": round(full_metrics[s][j].ci_lb, 4)})

        # 5-fold cross-fitted held-out MAE for R5 at this tau (English sources)
        maes, abstains = [], 0
        for s in ENGLISH_SOURCES:
            records = per_source_records[s]
            for cal_idx, ev_idx in KFold(5, shuffle=True, random_state=42).split(records):
                cal = [records[i] for i in cal_idx]
                ev = [records[i] for i in ev_idx]
                sel = select_r5(calibrate(cal), tau)
                if sel is None:
                    abstains += 1
                    continue
                m = fold_mae(ev, sel)
                if m is not None:
                    maes.append(m)
        info["r5_heldout"] = {
            "mean_mae": round(sum(maes) / len(maes), 4) if maes else None,
            "n_folds_scored": len(maes),
            "abstain_folds": abstains,
            "total_folds": 25,
        }
        out["by_tau"][f"{tau:.2f}"] = info

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # -------- console summary --------
    print(f"{'tau':>5} | " + " ".join(f"{s[:6]:>10}" for s in SOURCES) + " | heldout MAE (abst.)")
    for tau in TAUS:
        info = out["by_tau"][f"{tau:.2f}"]
        cells = []
        for s in SOURCES:
            ps = info["per_source"][s]
            sel = ps["r5_selected"] or "ABSTAIN"
            cells.append(f"{ps['n_pass']}|{sel[:7]:>7}")
        h = info["r5_heldout"]
        print(f"{tau:>5.2f} | " + " ".join(f"{c:>10}" for c in cells) +
              f" | {h['mean_mae']} ({h['abstain_folds']}/{h['total_folds']})")
    print("\nStatus flips vs tau=0.50:")
    for tau in TAUS:
        flips = out["by_tau"][f"{tau:.2f}"]["flips_vs_050"]
        names = ", ".join(f"{f['source']}:{f['judge']}({f['at_050']}->{f['at_tau']})" for f in flips)
        print(f"  tau={tau:.2f}: {len(flips)} flips  {names}")
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
