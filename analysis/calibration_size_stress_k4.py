"""Calibration-set-size stress test on the original K=4 pool.

The paper's stress-test figure is scoped to the four judges of the original pool
(Qwen-7B, Llama-3.2-3B, Mistral-7B, Llama Guard 3-1B). An earlier version of this
analysis omitted Llama Guard 3-1B on the grounds that its unsafe-recall counts on
this source match Qwen-7B's, so its bootstrap curve would coincide. That reasoning
does not hold: the two judges agree in aggregate (29 true positives each on the 42
unsafe records) but disagree on 10 of those records individually, so resampling
moves them differently. Llama Guard 3-1B is included explicitly here.

For each calibration size n, we resample n records with replacement (B replicates),
recompute each judge's Stage-1 status, and apply the full protocol. A LowConfidence
fallback returns a judge but is an abstention from screened reporting, so only a
SCREENED verdict counts as a certified selection.

Run:
    python calibration_size_stress_k4.py --bootstrap 1000
"""

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protocol.data import calibrate, default_root, load_source, results_dir
from protocol.judge_calibration import select_primary_judge

K4_POOL = ["qwen7b", "llama3b", "mistral7b", "llamaguard3_1b"]
SIZES = [50, 100, 150, 200, 300]


def run(root: str, sizes: list, n_bootstrap: int, seed: int) -> dict:
    records = load_source("harmful", root)
    rng = np.random.default_rng(seed)
    out = {}
    for n in sizes:
        pass_counts = Counter()
        selections = Counter()
        n_abstain = 0
        for _ in range(n_bootstrap):
            idx = rng.integers(0, len(records), size=n)
            sub = [records[i] for i in idx]
            metrics = calibrate(sub, judges=K4_POOL)
            for judge in K4_POOL:
                m = metrics.get(judge)
                if m is not None and m.stage1 == "PASS":
                    pass_counts[judge] += 1
            decision = select_primary_judge(
                {j: m for j, m in metrics.items() if m is not None})
            if decision["status"] != "SELECT":
                n_abstain += 1
            else:
                selections[decision["selected"]] += 1

        emitted = n_bootstrap - n_abstain
        modal, modal_count = (selections.most_common(1)[0]
                              if selections else (None, 0))
        out[str(n)] = {
            "n_bootstrap": n_bootstrap,
            "stage1_pass_freq": {j: pass_counts[j] / n_bootstrap for j in K4_POOL},
            "fallback_freq": n_abstain / n_bootstrap,
            "emit_freq": emitted / n_bootstrap,
            "selection_stability": (modal_count / emitted) if emitted else 0.0,
            "modal_selection": modal,
            "n_distinct_selected": len(selections),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=default_root())
    parser.add_argument("--out", default=None)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = run(args.root, SIZES, args.bootstrap, args.seed)

    print(f"K=4 pool: {', '.join(K4_POOL)}   bootstrap={args.bootstrap}\n")
    header = f"{'n':>5}" + "".join(f"{j:>16}" for j in K4_POOL) + f"{'abstain':>10}{'stability':>11}"
    print(header)
    for n in SIZES:
        row = out[str(n)]
        cells = "".join(f"{row['stage1_pass_freq'][j]:>16.3f}" for j in K4_POOL)
        print(f"{n:>5}{cells}{row['fallback_freq']:>10.3f}{row['selection_stability']:>11.3f}")

    dest = args.out or os.path.join(results_dir(args.root), "calibration_size_stress_k4.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2)
    print(f"\nSaved {dest}")


if __name__ == "__main__":
    main()
