"""Analyze the multilingual-judge GPU runs (Reviewer 1, Q5/Q11).

Computes Stage-1 calibration metrics for each new judge/variant against the
human-adjudicated labels, under two mappings of three-way outputs:
  strict        UNSAFE only counts as unsafe (CONTROVERSIAL -> safe)
  conservative  UNSAFE or CONTROVERSIAL counts as unsafe

Inputs:  revise/gpu_results/<judge>_<source>.json
Output:  revise/experiments/results/multilingual_judges.json

Run:  python analyze_multilingual.py
"""
import json
import os
import sys
from glob import glob

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

from protocol.data import SOURCES, load_source
from protocol.judge_calibration import compute_metrics, metrics_to_dict

ROOT = os.environ.get("DISTILL_ROOT", os.path.abspath(os.path.join(HERE, "..", "..")))
GPU_DIR = os.path.join(HERE, "..", "gpu_results")
OUT = os.path.join(HERE, "results", "multilingual_judges.json")


def load_predictions(path):
    data = json.load(open(path, encoding="utf-8"))
    return data["judge"], data["source"], {
        p["annotation_id"]: p["label"] for p in data["predictions"]}


def evaluate(judge_name, source, pred_map, mapping):
    records = load_source(source, ROOT)
    human, judged = [], []
    n_controversial = 0
    for r in records:
        label = pred_map.get(r["annotation_id"])
        if r["true_unsafe"] is None or label is None:
            continue
        if label == "CONTROVERSIAL":
            n_controversial += 1
        if mapping == "strict":
            unsafe = label == "UNSAFE"
        else:
            unsafe = label in ("UNSAFE", "CONTROVERSIAL")
        human.append(r["true_unsafe"])
        judged.append(unsafe)
    if not human:
        return None
    m = metrics_to_dict(compute_metrics(human, judged))
    m["n_controversial"] = n_controversial
    return m


def main():
    paths = sorted(glob(os.path.join(GPU_DIR, "*.json")))
    if not paths:
        raise SystemExit(
            f"No judge prediction files found in {GPU_DIR}. Refusing to overwrite "
            f"{OUT} with an empty result; bundle the zh-judge prediction JSONs first."
        )

    results = {}
    for path in paths:
        judge, source, preds = load_predictions(path)
        results.setdefault(judge, {})
        results[judge][source] = {
            mapping: evaluate(judge, source, preds, mapping)
            for mapping in ("strict", "conservative")
        }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    for judge, by_source in results.items():
        print(f"\n=== {judge} ===")
        print(f"  {'source':<13}{'map':<14}{'n':>5}{'n_uns':>6}{'sens':>7}{'spec':>7}"
              f"{'BA':>7}{'CI_LB':>7}  stage1  (#controversial)")
        for source in SOURCES:
            if source not in by_source:
                continue
            for mapping in ("strict", "conservative"):
                m = by_source[source][mapping]
                if m is None:
                    continue
                print(f"  {source:<13}{mapping:<14}{m['n']:>5}{m['n_unsafe']:>6}"
                      f"{m['sens']:>7.3f}{m['spec']:>7.3f}{m['ba']:>7.3f}"
                      f"{m['ci_lb']:>7.3f}  {m['stage1']:<12} ({m['n_controversial']})")
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
