"""Confidence-interval filter vs. point-estimate filter (Stage 1).

Compares the paper's Stage-1 rule (Clopper-Pearson lower bound on unsafe recall
>= tau_min, with an UNDERPOWERED guard at n_unsafe < nmin) against a naive
point-estimate rule (sensitivity >= tau_min). Every disagreement is a cell where
the point estimate would admit a judge that the CI rule rejects or flags as
under-powered, i.e. exactly the over-confident admissions the CI filter prevents.

Reads ``analysis_results/k9_calibration.json``.

Run:
    python ci_vs_point_estimate_ablation.py
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protocol.data import default_root, results_dir
from protocol.judge_calibration import NMIN_UNSAFE, TAU_MIN


def run(calibration: dict, tau_min: float, nmin: int) -> dict:
    cells = []
    for source, block in calibration["calibration"].items():
        for judge, m in block["judges"].items():
            if m is None:
                continue
            ci_pass = m["n_unsafe"] >= nmin and m["ci_lb"] >= tau_min
            point_pass = m["sens"] >= tau_min
            cells.append(
                {
                    "source": source,
                    "judge": judge,
                    "n_unsafe": m["n_unsafe"],
                    "sens": m["sens"],
                    "ci_lb": m["ci_lb"],
                    "ci_pass": ci_pass,
                    "point_pass": point_pass,
                    "underpowered": m["n_unsafe"] < nmin,
                }
            )

    flips = [c for c in cells if c["point_pass"] and not c["ci_pass"]]
    return {
        "tau_min": tau_min,
        "nmin_unsafe": nmin,
        "total_cells": len(cells),
        "ci_pass": sum(c["ci_pass"] for c in cells),
        "point_pass": sum(c["point_pass"] for c in cells),
        "point_pass_ci_reject": len(flips),
        "flips": flips,
    }


def print_summary(summary: dict) -> None:
    print(f"Total (source, judge) cells : {summary['total_cells']}")
    print(f"CI filter admits            : {summary['ci_pass']}")
    print(f"Point estimate admits       : {summary['point_pass']}")
    print(f"Point admits, CI rejects    : {summary['point_pass_ci_reject']}")
    print("\nCells where the point estimate would over-admit:")
    print(f"  {'source':<12}{'judge':<16}{'n_uns':>6}{'sens':>7}{'CI_LB':>7}  flag")
    for c in summary["flips"]:
        flag = "underpowered" if c["underpowered"] else "CI below tau_min"
        print(f"  {c['source']:<12}{c['judge']:<16}{c['n_unsafe']:>6}{c['sens']:>7.3f}{c['ci_lb']:>7.3f}  {flag}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=default_root())
    parser.add_argument("--calibration", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--tau-min", type=float, default=TAU_MIN)
    parser.add_argument("--nmin-unsafe", type=int, default=NMIN_UNSAFE)
    args = parser.parse_args()

    cal_path = args.calibration or os.path.join(results_dir(args.root), "k9_calibration.json")
    with open(cal_path, encoding="utf-8") as handle:
        calibration = json.load(handle)

    summary = run(calibration, args.tau_min, args.nmin_unsafe)
    print_summary(summary)

    out_path = args.out or os.path.join(results_dir(args.root), "ci_vs_point_ablation.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
