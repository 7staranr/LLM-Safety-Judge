"""Selection-rule ablation (R1-R5) across the five English prompt sources.

Each rule picks one judge per source from the calibration metrics:

* R1  max balanced accuracy
* R2  max MCC
* R3  max unsafe recall (point estimate)
* R4  max safe recall (specificity)
* R5  the paper protocol: keep judges whose Clopper-Pearson lower bound on
      unsafe recall clears tau_min, then break ties on balanced accuracy;
      abstain when none qualify.

The headline result is that R3 and R4 can select a judge that fails Stage 1
(an under-calibrated, low-recall judge), while R5 never does -- it abstains
instead.

Reads ``analysis_results/k9_calibration.json`` (produced by
compute_k9_calibration.py).

Run:
    python cross_source_protocol_ablation.py
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protocol.data import ENGLISH_SOURCES, default_root, results_dir
from protocol.judge_calibration import NMIN_UNSAFE, TAU_MIN

RULES = ["R1_max_ba", "R2_max_mcc", "R3_max_unsafe_recall", "R4_max_safe_recall", "R5_protocol"]


def _effective_stage1(m: dict, tau_min: float, nmin: int) -> str:
    """Stage-1 status recomputed at the active tau_min (the stored status uses
    the default), so that --tau-min propagates consistently."""
    if m["n_unsafe"] < nmin:
        return "UNDERPOWERED"
    return "PASS" if m["ci_lb"] >= tau_min else "FAIL"


def _argmax(metrics: dict, key: str):
    """Judge with the highest metric ``key``; ties broken alphabetically."""
    return sorted(metrics, key=lambda j: (-metrics[j][key], j))[0]


def _select(metrics: dict, stage1: dict, rule: str):
    """Return (selected_judge or None, abstained)."""
    if rule == "R5_protocol":
        passed = {j: m for j, m in metrics.items() if stage1[j] == "PASS"}
        if not passed:
            return None, True
        return _argmax(passed, "ba"), False

    key = {
        "R1_max_ba": "ba",
        "R2_max_mcc": "mcc",
        "R3_max_unsafe_recall": "sens",
        "R4_max_safe_recall": "spec",
    }[rule]
    return _argmax(metrics, key), False


def run(calibration: dict, tau_min: float, nmin: int) -> dict:
    results = {}
    for source in ENGLISH_SOURCES:
        metrics = {
            judge: m
            for judge, m in calibration["calibration"][source]["judges"].items()
            if m is not None
        }
        stage1 = {j: _effective_stage1(m, tau_min, nmin) for j, m in metrics.items()}
        results[source] = {}
        for rule in RULES:
            selected, abstained = _select(metrics, stage1, rule)
            if selected is None:
                status = "LOW_CONFIDENCE" if "UNDERPOWERED" in stage1.values() else "ABORT"
                results[source][rule] = {
                    "selected": None, "abstained": True, "status": status,
                    "stage1_fail": False, "stage1_nonpass": False,
                }
            else:
                m = metrics[selected]
                s1 = stage1[selected]
                results[source][rule] = {
                    "selected": selected,
                    "abstained": False,
                    "stage1": s1,
                    "stage1_fail": s1 == "FAIL",
                    "stage1_nonpass": s1 != "PASS",
                    "sens": m["sens"],
                    "ba": m["ba"],
                    "ci_lb": m["ci_lb"],
                }
    return results


def print_table(results: dict) -> None:
    print(f"{'source':<12}{'rule':<22}{'selected':<16}{'sens':>6}{'BA':>6}{'CI_LB':>7}  stage1")
    for source in results:
        for rule in RULES:
            info = results[source][rule]
            if info["selected"] is None:
                print(f"{source:<12}{rule:<22}{'(abstain: ' + info['status'] + ')':<24}")
            else:
                print(
                    f"{source:<12}{rule:<22}{info['selected']:<16}"
                    f"{info['sens']:>6.3f}{info['ba']:>6.3f}{info['ci_lb']:>7.3f}  {info['stage1']}"
                )
        print()

    print("Non-PASS judge selected, per rule (lower is better):")
    for rule in RULES:
        n_fail = sum(1 for s in results if results[s][rule]["stage1_fail"])
        n_nonpass = sum(1 for s in results if results[s][rule]["stage1_nonpass"])
        n_abstain = sum(1 for s in results if results[s][rule]["abstained"])
        print(f"  {rule:<22}FAIL {n_fail}/{len(results)}  non-PASS {n_nonpass}/{len(results)}  "
              f"(abstained on {n_abstain})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=default_root())
    parser.add_argument("--calibration", default=None, help="path to k9_calibration.json")
    parser.add_argument("--out", default=None)
    parser.add_argument("--tau-min", type=float, default=TAU_MIN)
    parser.add_argument("--nmin-unsafe", type=int, default=NMIN_UNSAFE)
    args = parser.parse_args()

    cal_path = args.calibration or os.path.join(results_dir(args.root), "k9_calibration.json")
    with open(cal_path, encoding="utf-8") as handle:
        calibration = json.load(handle)

    results = run(calibration, args.tau_min, args.nmin_unsafe)
    print_table(results)

    out_path = args.out or os.path.join(results_dir(args.root), "cross_source_protocol_ablation.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
