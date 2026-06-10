"""Robustness of Stage-1 decisions to the label-aggregation rule.

The released labels use a safety-conservative adjudication (disagreements ->
unsafe). This re-runs Stage 1 under five label rules -- adjudicated, annotator 1
only, annotator 2 only, disagreement -> unsafe, disagreement -> safe -- and
reports how few (source, judge) cells change their PASS/FAIL status, i.e. how
robust the cross-source pattern is to reasonable labelling choices.

Runs on the five English sources, which carry two independent annotations.

Run:
    python label_noise_robustness.py
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protocol.data import ENGLISH_SOURCES, JUDGES, calibrate, default_root, load_source, results_dir

RULES = ["adjudicated", "annotator_1", "annotator_2", "disagree_unsafe", "disagree_safe"]


def run(root: str) -> dict:
    result = {}
    for source in ENGLISH_SOURCES:
        per_rule = {}
        for rule in RULES:
            records = load_source(source, root, label_rule=rule)
            metrics = calibrate(records)
            per_rule[rule] = {
                judge: (
                    {"n_unsafe": m.n_unsafe, "sens": round(m.sens, 4),
                     "ci_lb": round(m.ci_lb, 4), "stage1": m.stage1}
                    if m is not None else None
                )
                for judge, m in metrics.items()
            }
        result[source] = per_rule
    return result


def summarize(result: dict) -> dict:
    """Count (source, judge) cells whose PASS/FAIL flips vs. the adjudicated rule."""
    flips = []
    for source, per_rule in result.items():
        base = per_rule["adjudicated"]
        for rule in RULES[1:]:
            for judge in JUDGES:
                b, a = base.get(judge), per_rule[rule].get(judge)
                if not b or not a:
                    continue
                if (b["stage1"] == "PASS") != (a["stage1"] == "PASS"):
                    flips.append(
                        {"source": source, "judge": judge, "rule": rule,
                         "adjudicated": b["stage1"], "alt": a["stage1"],
                         "adjudicated_ci_lb": b["ci_lb"], "alt_ci_lb": a["ci_lb"]}
                    )
    n_cells = sum(
        1
        for source, per_rule in result.items()
        for rule in RULES[1:]
        for judge in JUDGES
        if per_rule["adjudicated"].get(judge) and per_rule[rule].get(judge)
    )
    return {"n_alternative_cells": n_cells, "n_flips": len(flips), "flips": flips}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=default_root())
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    result = run(args.root)
    summary = summarize(result)

    print(f"Alternative-rule (source, judge) cells : {summary['n_alternative_cells']}")
    print(f"Cells that flip PASS/FAIL vs adjudicated: {summary['n_flips']}")
    for f in summary["flips"]:
        print(
            f"  {f['source']:<12}{f['judge']:<16}{f['rule']:<16}"
            f"{f['adjudicated']} ({f['adjudicated_ci_lb']:.3f}) -> {f['alt']} ({f['alt_ci_lb']:.3f})"
        )

    out_path = args.out or os.path.join(results_dir(args.root), "label_noise_robustness.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump({"per_source": result, "summary": summary}, handle, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
