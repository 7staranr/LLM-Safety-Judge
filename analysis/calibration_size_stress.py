"""Calibration-budget stress test of the two-stage protocol.

Bootstrap-subsamples the harmful calibration set down to a range of sizes and,
at each size, re-runs the full protocol (CI filter + balanced-accuracy tie-break,
or abstention). It reports how often each judge clears Stage 1, how stable the
selected judge is, and how often the protocol abstains -- showing that with a
small calibration budget the protocol abstains rather than committing to an
under-evidenced judge.

Runs on the harmful source (K = 9 judges).

Run:
    python calibration_size_stress.py --bootstrap 1000
"""

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protocol.data import JUDGES, calibrate, default_root, load_source, results_dir
from protocol.judge_calibration import select_primary_judge

SIZES = [50, 100, 150, 200, 300]


def run(records: list, sizes: list, n_bootstrap: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n_total = len(records)
    out = {"n_total": n_total, "n_bootstrap": n_bootstrap, "by_size": {}}

    for size in sizes:
        pass_counts = Counter()
        selections = Counter()
        n_abstain = 0
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n_total, size=min(size, n_total))
            sample = [records[i] for i in idx]
            metrics = calibrate(sample)
            for judge in JUDGES:
                m = metrics.get(judge)
                if m is not None and m.stage1 == "PASS":
                    pass_counts[judge] += 1
            decision = select_primary_judge({j: m for j, m in metrics.items() if m is not None})
            if decision["selected"] is None:
                n_abstain += 1
            else:
                selections[decision["selected"]] += 1

        n_emit = n_bootstrap - n_abstain
        modal, modal_n = (selections.most_common(1)[0] if selections else (None, 0))
        out["by_size"][str(size)] = {
            "stage1_pass_freq": {j: pass_counts[j] / n_bootstrap for j in JUDGES},
            "abstain_rate": n_abstain / n_bootstrap,
            "modal_selection": modal,
            "selection_stability": (modal_n / n_emit) if n_emit else 0.0,
            "n_distinct_selected": len(selections),
        }
    return out


def print_summary(out: dict) -> None:
    print(f"harmful source, n_total={out['n_total']}, bootstrap={out['n_bootstrap']}\n")
    print(f"  {'n':>4}{'abstain':>9}{'modal judge':>16}{'stability':>11}{'#distinct':>11}")
    for size, info in out["by_size"].items():
        print(
            f"  {size:>4}{info['abstain_rate']:>9.3f}{str(info['modal_selection']):>16}"
            f"{info['selection_stability']:>11.3f}{info['n_distinct_selected']:>11}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=default_root())
    parser.add_argument("--out", default=None)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    records = load_source("harmful", args.root)
    out = run(records, SIZES, args.bootstrap, args.seed)
    print_summary(out)

    out_path = args.out or os.path.join(results_dir(args.root), "calibration_size_stress.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
