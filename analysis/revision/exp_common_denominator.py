"""Common-denominator deployment audit and the estimands it separates.

Reviewer finding (2026-07-25): the deployment-audit table compared judges on
judge-specific effective denominators, because outputs that did not parse as
SAFE / UNSAFE were dropped per judge. Phi-3-medium-4k scored 90 of the 100 pairs
in the 1:0 condition and WildGuard 90, while Qwen-7B scored all 100, so the
reported rates were not measured on the same responses. The same review noted
that the paper's "six-fold" headline conflates two different quantities.

This script recomputes the audit on the intersection of pairs that EVERY judge
scored, so all judges share one denominator, and reports the two estimands
separately:

  * judge-to-judge spread: max judge ASR / min judge ASR, i.e. how much the
    reported number moves purely by choosing a different judge;
  * underestimation factor: human ASR / lowest judge ASR, i.e. how far the worst
    judge falls below the human ground truth.

It also reports the per-judge non-parse counts, so n_effective can be stated in
the table.

Run: python exp_common_denominator.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from protocol.data import load_source

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("DISTILL_ROOT",
                      os.path.abspath(os.path.join(HERE, "..", "..")))
OUT = os.path.join(HERE, "results", "common_denominator_audit.json")

K9 = ["mistral7b", "shieldgemma_9b", "llama31_8b", "llama3b", "gemma2_9b",
      "wildguard", "qwen7b", "phi3_medium", "llamaguard3_1b"]
RATIOS = ["1to0", "2to1", "1to1"]


def main():
    records = load_source("harmful", ROOT)
    out = {"conditions": {}, "per_judge_nonparse": {}}

    # per-judge non-parse counts over the whole source
    for j in K9:
        scored = sum(1 for r in records if r["preds"].get(j) is not None)
        out["per_judge_nonparse"][j] = {
            "n_scored": scored, "n_total": len(records),
            "n_nonparse": len(records) - scored,
            "nonparse_rate": round((len(records) - scored) / len(records), 4),
        }

    for ratio in RATIOS:
        cond = [r for r in records if r["ratio"] == ratio]
        # intersection: pairs every judge scored and a human labelled
        common = [r for r in cond
                  if r["true_unsafe"] is not None
                  and all(r["preds"].get(j) is not None for j in K9)]
        per_judge_own, per_judge_common = {}, {}
        for j in K9:
            own = [r for r in cond if r["preds"].get(j) is not None]
            per_judge_own[j] = {
                "n": len(own),
                "asr": round(sum(1 for r in own if r["preds"][j]) / len(own), 4) if own else None,
            }
            per_judge_common[j] = round(
                sum(1 for r in common if r["preds"][j]) / len(common), 4) if common else None

        human = round(sum(1 for r in common if r["true_unsafe"]) / len(common), 4) if common else None
        vals = [v for v in per_judge_common.values() if v is not None]
        lo, hi = min(vals), max(vals)
        lo_j = min(per_judge_common, key=lambda k: per_judge_common[k])
        hi_j = max(per_judge_common, key=lambda k: per_judge_common[k])

        out["conditions"][ratio] = {
            "n_common": len(common),
            "human_asr": human,
            "per_judge_common_denominator": per_judge_common,
            "per_judge_own_denominator": per_judge_own,
            "spread_max_over_min": round(hi / lo, 2) if lo else None,
            "spread_judges": [hi_j, lo_j],
            "underestimation_human_over_min": round(human / lo, 2) if lo else None,
        }

    c = out["conditions"]["1to0"]
    print(f"1:0 condition, common denominator n = {c['n_common']} "
          f"(pairs every judge scored)\n")
    print(f"  {'judge':<18}{'own-n ASR':>12}{'own n':>8}{'common ASR':>13}")
    for j in K9:
        o = c["per_judge_own_denominator"][j]
        print(f"  {j:<18}{o['asr']:>12.4f}{o['n']:>8}"
              f"{c['per_judge_common_denominator'][j]:>13.4f}")
    print(f"  {'human truth':<18}{'':>12}{'':>8}{c['human_asr']:>13.4f}")
    print(f"\n  judge-to-judge spread (common denominator): "
          f"{c['spread_max_over_min']}x  ({c['spread_judges'][0]} / {c['spread_judges'][1]})")
    print(f"  underestimation vs human truth: {c['underestimation_human_over_min']}x")

    print("\nNon-parse rates over the harmful source:")
    for j in K9:
        n = out["per_judge_nonparse"][j]
        if n["n_nonparse"]:
            print(f"  {j:<18}{n['n_nonparse']:>4} / {n['n_total']}  "
                  f"({n['nonparse_rate']*100:.1f}%)")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=2)
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
