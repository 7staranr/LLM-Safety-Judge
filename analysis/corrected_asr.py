"""Diagnostic Rogan-Gladen ASR correction for Stage-1-passing judges.

For each judge that passes Stage 1 on the harmful source, this reports the raw
per-condition ASR (the judge's unsafe rate per ``source_ratio``) alongside the
Rogan-Gladen corrected ASR, which adjusts for the judge's imperfect sensitivity
and specificity. A percentile bootstrap gives a confidence interval on the
corrected value. The correction is an optional diagnostic: it is only stable
when the judge has enough unsafe positives in the condition.

Runs on the harmful source (the only source with multiple SFT conditions).

Run:
    python corrected_asr.py --bootstrap 1000
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protocol.data import RATIOS, calibrate, default_root, load_source, results_dir
from protocol.judge_calibration import rogan_gladen


def _aligned(records: list, judge: str) -> list:
    """Pairs (human, judge) in a condition where both labels are present."""
    return [(r["true_unsafe"], r["preds"].get(judge)) for r in records
            if r["true_unsafe"] is not None and r["preds"].get(judge) is not None]


def _percentile_ci(values: list) -> dict:
    if not values:
        return {"mean": None, "ci_lo": None, "ci_hi": None}
    return {
        "mean": float(np.mean(values)),
        "ci_lo": float(np.percentile(values, 2.5)),
        "ci_hi": float(np.percentile(values, 97.5)),
    }


def _bootstrap_conditions(records: list, judge: str, ratios: list, n_boot: int, rng) -> dict:
    """Resample the whole calibration set; recompute sens/spec on the resample and
    the per-condition corrected ASR from the same resample, propagating both
    sources of uncertainty coherently. Unscorable draws (no judged pairs in the
    condition) are skipped rather than counted as zero."""
    idx_all = np.arange(len(records))
    samples = {ratio: [] for ratio in ratios}
    for _ in range(n_boot):
        pick = rng.integers(0, len(records), size=len(records))
        resample = [records[i] for i in idx_all[pick]]
        m = calibrate(resample, judges=[judge])[judge]
        if m is None:
            continue
        for ratio in ratios:
            pairs = _aligned([r for r in resample if r["ratio"] == ratio], judge)
            if not pairs:
                continue
            raw = sum(1 for _, p in pairs if p) / len(pairs)
            corrected = rogan_gladen(raw, m.sens, m.spec)
            if corrected is not None:
                samples[ratio].append(corrected)
    return {ratio: _percentile_ci(vals) for ratio, vals in samples.items()}


def run(root: str, n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    records = load_source("harmful", root)
    metrics = calibrate(records)
    passing = [j for j, m in metrics.items() if m is not None and m.stage1 == "PASS"]

    out = {"source": "harmful", "passing_judges": passing, "by_judge": {}}
    for judge in passing:
        m = metrics[judge]
        boot = _bootstrap_conditions(records, judge, RATIOS, n_boot, rng)
        conditions = {}
        for ratio in RATIOS:
            cond = [r for r in records if r["ratio"] == ratio]
            pairs = _aligned(cond, judge)
            if not pairs:
                continue
            raw = sum(1 for _, p in pairs if p) / len(pairs)
            human = sum(1 for t, _ in pairs if t) / len(pairs)
            corrected = rogan_gladen(raw, m.sens, m.spec)
            conditions[ratio] = {
                "n": len(cond),
                "n_judged": len(pairs),
                "human_asr": round(human, 4),
                "raw_asr": round(raw, 4),
                "corrected_asr": (round(corrected, 4) if corrected is not None else None),
                "corrected_bootstrap": boot[ratio],
            }
        out["by_judge"][judge] = {
            "sens": round(m.sens, 4), "spec": round(m.spec, 4), "conditions": conditions,
        }
    return out


def print_summary(out: dict) -> None:
    print(f"harmful source, Stage-1-passing judges: {', '.join(out['passing_judges'])}\n")
    for judge, block in out["by_judge"].items():
        print(f"[{judge}]  sens={block['sens']:.3f}  spec={block['spec']:.3f}")
        print(f"  {'ratio':<8}{'n':>5}{'judged':>7}{'human':>8}{'raw':>8}{'corrected':>11}{'95% CI':>20}")
        for ratio, c in block["conditions"].items():
            boot = c["corrected_bootstrap"]
            ci = (f"[{boot['ci_lo']:.3f},{boot['ci_hi']:.3f}]"
                  if boot["ci_lo"] is not None else "  -  ")
            corr = f"{c['corrected_asr']:.3f}" if c["corrected_asr"] is not None else "  -  "
            print(f"  {ratio:<8}{c['n']:>5}{c['n_judged']:>7}{c['human_asr']:>8.3f}"
                  f"{c['raw_asr']:>8.3f}{corr:>11}{ci:>20}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=default_root())
    parser.add_argument("--out", default=None)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = run(args.root, args.bootstrap, args.seed)
    print_summary(out)

    out_path = args.out or os.path.join(results_dir(args.root), "corrected_asr.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
