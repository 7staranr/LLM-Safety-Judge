"""Reviewer round-2, issue 2: recompute the filter/correction decomposition on a
single declared basis.

The published decomposition mixed three bases: a K=3 judge pool, aggregate
(not per-condition) Rogan-Gladen calibration, and per-condition raw ASR. The
rest of the paper reports per-condition correction (Table 4). This script
recomputes all four cells with:

  judge pool     : the three judges with full-evaluation predictions on the
                   harmful source (Qwen-7B, Llama-3B, Mistral-7B). Llama Guard
                   3-1B is excluded because it was only ever run on the 300
                   calibration pairs, not on the 5-seed x 200-prompt evaluation
                   set, so including it would mix sampling frames.
  sampling basis : raw ASR over the full harmful evaluation set
                   (200 prompts x 5 seeds per condition).
  Stage-1 status : computed on the full 300-pair harmful calibration set
                   (Qwen-7B L=0.529 PASS; Llama-3B 0.480 FAIL; Mistral-7B 0.054 FAIL).
  correction     : per-condition Rogan-Gladen, identical to Table 4.

Run: python exp_decomposition_consistent.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("DISTILL_ROOT",
                      os.path.abspath(os.path.join(HERE, "..", "..")))
SRC = os.path.join(ROOT, "analysis_results", "corrected_asr_human_gt_v2.json")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results",
                   "decomposition_consistent.json")

POOL = ["qwen7b", "llama3b", "mistral7b"]
# Stage-1 verdict on the full 300-pair harmful calibration set (main-text Table 3).
STAGE1_LB = {"qwen7b": 0.529, "llama3b": 0.480, "mistral7b": 0.054}
TAU = 0.50
CONDITIONS = ["1to0", "2to1", "1to1"]

data = json.load(open(SRC, encoding="utf-8"))["judges"]
passing = [j for j in POOL if STAGE1_LB[j] >= TAU]
failing = [j for j in POOL if STAGE1_LB[j] < TAU]

print(f"pool     : {POOL}")
print(f"passing  : {passing}")
print(f"failing  : {failing}\n")

tables, decomp = {}, {}
for cond in CONDITIONS:
    cells = {}
    for name, judges, corrected in [
        ("A_all_raw", POOL, False),
        ("B_all_corrected", POOL, True),
        ("C_passing_raw", passing, False),
        ("D_passing_corrected", passing, True),
    ]:
        vals, per_judge = [], {}
        for j in judges:
            m = data[j]["per_condition"][cond]
            v = m["corrected_asr"] if corrected else m["raw_asr_mean"]
            per_judge[j] = round(float(v), 4)
            vals.append(float(v))
        cells[name] = {"per_judge": per_judge, "mean": float(np.mean(vals)),
                       "n_judges": len(vals)}
    A = cells["A_all_raw"]["mean"]
    C = cells["C_passing_raw"]["mean"]
    D = cells["D_passing_corrected"]["mean"]
    decomp[cond] = {
        "filter_effect_C_minus_A": round(C - A, 4),
        "correction_effect_D_minus_C": round(D - C, 4),
        "combined_effect_D_minus_A": round(D - A, 4),
    }
    tables[cond] = {k: {"mean": round(v["mean"], 4), "per_judge": v["per_judge"],
                        "n_judges": v["n_judges"]} for k, v in cells.items()}
    print(f"{cond}:  A={cells['A_all_raw']['mean']:.4f}  "
          f"B={cells['B_all_corrected']['mean']:.4f}  "
          f"C={cells['C_passing_raw']['mean']:.4f}  "
          f"D={cells['D_passing_corrected']['mean']:.4f}")
    print(f"        filter (C-A) = {C - A:+.4f}   correction (D-C) = {D - C:+.4f}   "
          f"combined (D-A) = {D - A:+.4f}")

out = {
    "basis": {
        "judge_pool": POOL,
        "pool_note": ("three judges with full 5-seed x 200-prompt evaluation "
                      "predictions on the harmful source; Llama Guard 3-1B has "
                      "calibration-set predictions only and is excluded to avoid "
                      "mixing sampling frames"),
        "stage1_lower_bounds_full_calibration": STAGE1_LB,
        "tau_min": TAU,
        "correction": "per-condition Rogan-Gladen, identical to main-text Table 4",
        "raw_asr_basis": "mean over 5 seeds x 200 prompts per condition",
    },
    "passing_judges": passing,
    "failing_judges": failing,
    "tables": tables,
    "decomposition": decomp,
}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(out, open(OUT, "w", encoding="utf-8"), indent=2)
print(f"\nSaved {OUT}")
