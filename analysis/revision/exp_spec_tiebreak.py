"""Reviewer 1, Q3: Stage-2 tie-break by specificity instead of balanced accuracy.

Compares three Stage-2 variants among Stage-1 passers on each English source:
  * R5-BA    argmax balanced accuracy (the paper protocol)
  * R5-Spec  argmax specificity (minimise false positives)
  * R5-Sens  argmax sensitivity (maximise unsafe recall), for completeness

Reports full-data selections and 5-fold cross-fitted held-out MAE plus the
SIGNED per-condition error (judge ASR - human ASR) to show the direction of
the bias each tie-break induces.

Run:  python exp_spec_tiebreak.py
"""
import json
import os
import sys

from sklearn.model_selection import KFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

from protocol.data import ENGLISH_SOURCES, calibrate, load_source

ROOT = os.environ.get("DISTILL_ROOT", os.path.abspath(os.path.join(HERE, "..", "..")))
OUT = os.path.join(HERE, "results", "spec_tiebreak.json")
VARIANTS = {"R5_BA": "ba", "R5_Spec": "spec", "R5_Sens": "sens"}


def select(metrics, attr):
    passed = {j: m for j, m in metrics.items() if m is not None and m.stage1 == "PASS"}
    if not passed:
        return None
    return sorted(passed, key=lambda j: (-getattr(passed[j], attr), j))[0]


def condition_errors(group, judge):
    pairs = [(r["true_unsafe"], r["preds"].get(judge)) for r in group
             if r["true_unsafe"] is not None and r["preds"].get(judge) is not None]
    if not pairs:
        return None
    human = sum(1 for t, _ in pairs if t) / len(pairs)
    judged = sum(1 for _, p in pairs if p) / len(pairs)
    return judged - human  # signed


def fold_errors(records, judge):
    by_ratio = {}
    for r in records:
        by_ratio.setdefault(r["ratio"], []).append(r)
    signed = [e for e in (condition_errors(g, judge) for g in by_ratio.values()) if e is not None]
    if not signed:
        return None, None
    mae = sum(abs(e) for e in signed) / len(signed)
    bias = sum(signed) / len(signed)
    return mae, bias


def fold_fpr(records, judge):
    """False positives of the selected judge on the held-out fold's safe pairs.

    Reviewer 1, Q3 asks for the false-positive quantities directly rather than
    inferred from MAE. Returns (n_safe, false_positives, fpr) where fpr = FP/N_safe
    = 1 - specificity, computed on the human-labelled safe pairs of the fold.
    """
    safe = [(r["true_unsafe"], r["preds"].get(judge)) for r in records
            if r["true_unsafe"] is False and r["preds"].get(judge) is not None]
    if not safe:
        return None
    fp = sum(1 for _, p in safe if p)
    return len(safe), fp, fp / len(safe)


def main():
    out = {"variants": list(VARIANTS), "per_source": {}, "aggregate": {}}
    agg = {v: {"mae": [], "bias": [], "abstain": 0, "fpr": [], "fp": 0, "n_safe": 0}
           for v in VARIANTS}

    for source in ENGLISH_SOURCES:
        records = load_source(source, ROOT)
        full = calibrate(records)
        row = {"full_data_selection": {v: select(full, attr) for v, attr in VARIANTS.items()},
               "kfold": {}}
        for variant, attr in VARIANTS.items():
            maes, biases, fprs = [], [], []
            fp_tot = nsafe_tot = 0
            for cal_idx, ev_idx in KFold(5, shuffle=True, random_state=42).split(records):
                cal = [records[i] for i in cal_idx]
                ev = [records[i] for i in ev_idx]
                sel = select(calibrate(cal), attr)
                if sel is None:
                    agg[variant]["abstain"] += 1
                    continue
                mae, bias = fold_errors(ev, sel)
                if mae is not None:
                    maes.append(mae)
                    biases.append(bias)
                fr = fold_fpr(ev, sel)
                if fr is not None:
                    n_safe, fp, fpr = fr
                    fprs.append(fpr)
                    fp_tot += fp
                    nsafe_tot += n_safe
            row["kfold"][variant] = {
                "mean_mae": round(sum(maes) / len(maes), 4) if maes else None,
                "mean_signed_bias": round(sum(biases) / len(biases), 4) if biases else None,
                "mean_fpr": round(sum(fprs) / len(fprs), 4) if fprs else None,
                "false_positives": fp_tot, "n_safe": nsafe_tot,
            }
            agg[variant]["mae"] += maes
            agg[variant]["bias"] += biases
            agg[variant]["fpr"] += fprs
            agg[variant]["fp"] += fp_tot
            agg[variant]["n_safe"] += nsafe_tot
        out["per_source"][source] = row

    for variant in VARIANTS:
        a = agg[variant]
        fpr_sd = None
        if len(a["fpr"]) > 1:
            m = sum(a["fpr"]) / len(a["fpr"])
            fpr_sd = round((sum((x - m) ** 2 for x in a["fpr"]) / (len(a["fpr"]) - 1)) ** 0.5, 4)
        out["aggregate"][variant] = {
            "mean_mae": round(sum(a["mae"]) / len(a["mae"]), 4) if a["mae"] else None,
            "mean_signed_bias": round(sum(a["bias"]) / len(a["bias"]), 4) if a["bias"] else None,
            "macro_fpr": round(sum(a["fpr"]) / len(a["fpr"]), 4) if a["fpr"] else None,
            "macro_fpr_sd": fpr_sd, "n_folds_fpr": len(a["fpr"]),
            "false_positives": a["fp"], "n_safe": a["n_safe"],
            "pooled_fpr": round(a["fp"] / a["n_safe"], 4) if a["n_safe"] else None,
            "abstain_folds": a["abstain"],
        }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"{'source':<12}" + "".join(f"{v:>22}" for v in VARIANTS))
    for source, row in out["per_source"].items():
        sels = "".join(f"{str(row['full_data_selection'][v]):>22}" for v in VARIANTS)
        print(f"{source:<12}{sels}")
    print("\nAggregate 5-fold held-out (25 folds):")
    for variant, a in out["aggregate"].items():
        print(f"  {variant:<8} MAE={a['mean_mae']}  signed bias={a['mean_signed_bias']:+.4f}  "
              f"macro-FPR={a['macro_fpr']}+-{a['macro_fpr_sd']} (pooled {a['pooled_fpr']}, "
              f"{a['false_positives']}/{a['n_safe']} FP)  abstains={a['abstain_folds']}")
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
