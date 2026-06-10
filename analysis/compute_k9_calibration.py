"""Per-source, per-judge calibration for the K = 9 candidate judges.

For every (source, judge) pair this computes sensitivity, specificity, balanced
accuracy, MCC, the Clopper-Pearson 95% lower bound on unsafe recall, and the
Stage-1 status (PASS / FAIL / UNDERPOWERED). The result is written to
``analysis_results/k9_calibration.json`` and is the single intermediate that the
protocol-ablation analyses consume.

A compact per-source table and the cross-source pass/reject pattern are printed.

Run:
    python compute_k9_calibration.py
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protocol.data import JUDGES, SOURCE_ABBR, SOURCES, calibrate, default_root, load_source, results_dir
from protocol.judge_calibration import ALPHA, NMIN_UNSAFE, TAU_MIN, metrics_to_dict


def build_calibration(root: str) -> dict:
    out = {
        "tau_min": TAU_MIN,
        "alpha": ALPHA,
        "nmin_unsafe": NMIN_UNSAFE,
        "judges": JUDGES,
        "sources": SOURCES,
        "calibration": {},
    }
    for source in SOURCES:
        records = load_source(source, root)
        metrics = calibrate(records)
        out["calibration"][source] = {
            "n_pairs": len(records),
            "judges": {
                judge: (metrics_to_dict(m) if m is not None else None)
                for judge, m in metrics.items()
            },
        }
    return out


def print_summary(result: dict) -> None:
    for source in result["sources"]:
        print(f"\n[{source}]  (n_pairs = {result['calibration'][source]['n_pairs']})")
        print(f"  {'judge':<16}{'n':>5}{'n_uns':>6}{'sens':>7}{'spec':>7}{'BA':>7}{'CI_LB':>7}  stage1")
        for judge in result["judges"]:
            m = result["calibration"][source]["judges"].get(judge)
            if m is None:
                print(f"  {judge:<16}{'-- no predictions --':>32}")
                continue
            print(
                f"  {judge:<16}{m['n']:>5}{m['n_unsafe']:>6}{m['sens']:>7.3f}"
                f"{m['spec']:>7.3f}{m['ba']:>7.3f}{m['ci_lb']:>7.3f}  {m['stage1']}"
            )

    print("\nCross-source pass/reject pattern (P=PASS, F=FAIL, u=UNDERPOWERED):")
    header = " ".join(SOURCE_ABBR[s].rjust(6) for s in result["sources"])
    print(f"  {'judge':<16}  {header}")
    code = {"PASS": "P", "FAIL": "F", "UNDERPOWERED": "u"}
    for judge in result["judges"]:
        cells = []
        for source in result["sources"]:
            m = result["calibration"][source]["judges"].get(judge)
            cells.append(code.get(m["stage1"], "?") if m else "?")
        n_pass = cells.count("P")
        marks = " ".join(c.rjust(6) for c in cells)
        print(f"  {judge:<16}  {marks}   ({n_pass}/{len(result['sources'])} PASS)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=default_root(), help="project root with the released inputs")
    parser.add_argument("--out", default=None, help="output JSON path")
    args = parser.parse_args()

    result = build_calibration(args.root)
    out_path = args.out or os.path.join(results_dir(args.root), "k9_calibration.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print_summary(result)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
