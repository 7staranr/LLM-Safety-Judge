"""Catastrophic-failure rate (CFR) under a small calibration budget.

A "catastrophic failure" is selecting a judge that fails Stage 1 on the full
calibration data (an under-calibrated, low-recall judge). This bootstrap-
subsamples the harmful calibration set to a range of sizes and measures, for
each selection rule, how often it commits such a judge. The two-stage protocol
(R5) abstains instead of committing, so its CFR stays near zero while the
metric-only rules spike at small budgets.

Runs on the harmful source (K = 9 judges).

Run:
    python catastrophic_failure_rate.py --bootstrap 1000
"""

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protocol.data import calibrate, default_root, load_source, results_dir
from protocol.judge_calibration import select_by_rule

SIZES = [50, 100, 150, 200, 300]
# Metric-only baselines compared against the protocol (R5).
RULES = ["R1_max_ba", "R2_max_mcc", "R3_max_unsafe_recall", "R5_protocol"]


def failing_judges(records: list) -> set:
    """Judges that fail Stage 1 on the full calibration data."""
    metrics = calibrate(records)
    return {j for j, m in metrics.items() if m is not None and m.stage1 == "FAIL"}


def run(records: list, sizes: list, n_bootstrap: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n_total = len(records)
    failing = failing_judges(records)

    out = {"n_total": n_total, "n_bootstrap": n_bootstrap,
           "failing_judges_full_data": sorted(failing), "by_size": {}}

    for size in sizes:
        cfr = Counter()
        emit = Counter()
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n_total, size=min(size, n_total))
            sample = [records[i] for i in idx]
            metrics = calibrate(sample)
            for rule in RULES:
                selected, abstained = select_by_rule(metrics, rule)
                if abstained:
                    continue
                emit[rule] += 1
                if selected in failing:
                    cfr[rule] += 1

        out["by_size"][str(size)] = {
            rule: {
                "cfr": cfr[rule] / n_bootstrap,
                "cfr_conditional_on_emit": (cfr[rule] / emit[rule]) if emit[rule] else 0.0,
                "emit_rate": emit[rule] / n_bootstrap,
            }
            for rule in RULES
        }
    return out


def print_summary(out: dict) -> None:
    rules = RULES
    print(f"harmful source, failing judges (full data): {', '.join(out['failing_judges_full_data'])}")
    print(f"bootstrap={out['n_bootstrap']}\n")
    print("CFR (unconditional) by calibration size:")
    print("  " + "size".rjust(5) + "".join(r.replace('_', ' ').rjust(22) for r in rules))
    for size, info in out["by_size"].items():
        print("  " + size.rjust(5) + "".join(f"{info[r]['cfr']:>22.3f}" for r in rules))


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

    out_path = args.out or os.path.join(results_dir(args.root), "catastrophic_failure_rate.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
