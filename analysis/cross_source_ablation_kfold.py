"""Cross-fitted MAE of the selection rules against human-estimated ASR.

For each source and each rule (R1-R5), this runs 5-fold cross-fitting: the rule
selects a judge on the calibration folds, and the selected judge's ASR is
compared against the human-estimated ASR on the held-out fold. ASR is measured
per SFT condition (``source_ratio``) and the per-fold error is averaged over
conditions. The reported aggregate (mean and worst-case MAE per rule) shows that
the protocol (R5) is competitive with the metric-only rules while never
committing a Stage-1-failing judge.

Runs on the five English sources, on the released human-labelled calibration
subset.

Run:
    python cross_source_ablation_kfold.py
"""

import argparse
import json
import os
import sys
from collections import Counter

from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protocol.data import ENGLISH_SOURCES, calibrate, default_root, load_source, results_dir
from protocol.judge_calibration import RULES, select_by_rule


def _condition_error(group: list, judge: str):
    """|judge ASR - human ASR| over the pairs in a condition where the judge has
    a prediction (human and judge rates use the same subset). None if empty."""
    pairs = [(r["true_unsafe"], r["preds"].get(judge)) for r in group
             if r["true_unsafe"] is not None and r["preds"].get(judge) is not None]
    if not pairs:
        return None
    human = sum(1 for t, _ in pairs if t) / len(pairs)
    judged = sum(1 for _, p in pairs if p) / len(pairs)
    return abs(judged - human)


def _fold_mae(eval_records: list, judge: str):
    """Mean over SFT conditions of |judge ASR - human ASR| on the held-out fold.
    None when no condition is scorable (the judge has no predictions in the fold),
    so the fold is excluded rather than counted as zero error."""
    by_ratio = {}
    for r in eval_records:
        by_ratio.setdefault(r["ratio"], []).append(r)
    errors = [e for e in (_condition_error(g, judge) for g in by_ratio.values()) if e is not None]
    return sum(errors) / len(errors) if errors else None


def run(root: str, n_splits: int, seed: int) -> dict:
    result = {}
    for source in ENGLISH_SOURCES:
        records = load_source(source, root)
        folds = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        per_rule = {rule: {"fold_mae": [], "selected": Counter(), "stage1_fail": 0,
                           "stage1_underpowered": 0, "abstain": 0, "unscorable": 0}
                    for rule in RULES}

        for cal_idx, eval_idx in folds.split(records):
            cal = [records[i] for i in cal_idx]
            ev = [records[i] for i in eval_idx]
            metrics = calibrate(cal)
            for rule in RULES:
                judge, abstained = select_by_rule(metrics, rule)
                if abstained:
                    per_rule[rule]["abstain"] += 1
                    continue
                per_rule[rule]["selected"][judge] += 1
                if metrics[judge].stage1 == "FAIL":
                    per_rule[rule]["stage1_fail"] += 1
                elif metrics[judge].stage1 == "UNDERPOWERED":
                    per_rule[rule]["stage1_underpowered"] += 1
                mae = _fold_mae(ev, judge)
                if mae is None:
                    per_rule[rule]["unscorable"] += 1
                else:
                    per_rule[rule]["fold_mae"].append(mae)

        result[source] = {
            "n_pairs": len(records),
            "rules": {
                rule: {
                    "mean_mae": (sum(d["fold_mae"]) / len(d["fold_mae"])) if d["fold_mae"] else None,
                    "max_mae": max(d["fold_mae"]) if d["fold_mae"] else None,
                    "modal_selection": (d["selected"].most_common(1)[0][0] if d["selected"] else None),
                    "stage1_fail_folds": d["stage1_fail"],
                    "stage1_underpowered_folds": d["stage1_underpowered"],
                    "abstain_folds": d["abstain"],
                    "unscorable_folds": d["unscorable"],
                }
                for rule, d in per_rule.items()
            },
        }
    return result


def print_summary(result: dict) -> None:
    for source, block in result.items():
        print(f"\n[{source}]  (n_pairs={block['n_pairs']})")
        print(f"  {'rule':<22}{'meanMAE':>9}{'maxMAE':>9}{'modal judge':>16}"
              f"{'S1-fail':>8}{'S1-u.p.':>8}{'abstain':>8}{'unscor':>8}")
        for rule, d in block["rules"].items():
            mean_mae = f"{d['mean_mae']:.3f}" if d["mean_mae"] is not None else "  -  "
            max_mae = f"{d['max_mae']:.3f}" if d["max_mae"] is not None else "  -  "
            print(
                f"  {rule:<22}{mean_mae:>9}{max_mae:>9}{str(d['modal_selection']):>16}"
                f"{d['stage1_fail_folds']:>8}{d['stage1_underpowered_folds']:>8}"
                f"{d['abstain_folds']:>8}{d['unscorable_folds']:>8}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=default_root())
    parser.add_argument("--out", default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = run(args.root, args.folds, args.seed)
    print_summary(result)

    out_path = args.out or os.path.join(results_dir(args.root), "cross_source_ablation_kfold.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
