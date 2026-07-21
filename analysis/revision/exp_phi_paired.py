"""Phi-3.5-mini cross-family inference under the design that was actually run.

Reviewer round-3, issue 1. The previous analyses (both the original Welch/seed
permutation and the two-way bootstrap that replaced it) treated the nine Phi
checkpoints as nine exchangeable units: three benign-only seeds versus six
"mixed" seeds. That misreads the design.

  * The same three seed identifiers {42, 123, 456} are reused across all three
    ratios, so seed is a blocking factor, not a replicate index.
  * The six mixed checkpoints are two fixed treatments (2:1 and 1:1) crossed
    with three seeds, not six exchangeable draws from one "mixed" condition.
  * All checkpoints are scored on the same 200 prompts.

The design is therefore a complete matched block: 3 seed blocks x 3 ratio
treatments, evaluated on a common prompt set. Treatments were assigned by
construction, not randomized within blocks, so no analysis here is
randomization-exact; any seed-level inference adds an assumption.

The estimand we care about is the benign-minus-mixed contrast, a within-block
quantity:

    D_{s,p} = Y_{1:0,s,p} - (1/2) ( Y_{2:1,s,p} + Y_{1:1,s,p} )
    D_s     = mean_p D_{s,p}                (block-level contrast)
    D_bar   = mean_s D_s                    (the estimate)

Assuming only that the block contrasts are independent and symmetric about zero
under the null, a sign-flip procedure over the 2^3 = 8 sign assignments has a
smallest attainable two-sided p-value of 2/8 = 0.25. That floor belongs to this
procedure, not to every possible analysis: a paired t-test, which instead assumes
the contrasts are normal, returns a much smaller number from the same three
points. With three blocks the assumption rather than the data separates those
answers, which is why we report the contrast and its direction and leave the
seed-level question open.

We also report a prompt-level cluster bootstrap of D_bar that keeps the seed
blocks intact (resample prompts, carry all three seeds), which quantifies
generalization to new prompts holding the realized seeds fixed. We do NOT
extrapolate to hypothetical seed and prompt budgets: the variance components of a
crossed design are not the invariant additive pieces such an extrapolation would
require, since a between-prompt term estimated after averaging seeds absorbs the
prompt-by-seed interaction, and a between-seed term estimated on one prompt set
is specific to it.

Run: python exp_phi_paired.py
"""
import glob
import itertools
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = r"D:\Programming\distill_project_1"
OUT = os.path.join(HERE, "results", "phi_paired.json")

RNG = np.random.default_rng(20260717)
B = 4000
JUDGE = "qwen7b"
SEEDS = [42, 123, 456]
RATIOS = ["1to0", "2to1", "1to1"]


def load():
    cells = {}
    for f in glob.glob(os.path.join(ROOT, "results_v4", f"v4_phi_*_{JUDGE}_labels.json")):
        m = re.search(r"phi_ratio_(\dto\d)_seed(\d+)", os.path.basename(f))
        if not m:
            continue
        data = json.load(open(f, encoding="utf-8"))
        vec = np.zeros(len(data))
        for d in data:
            vec[d["prompt_idx"]] = 1.0 if d["is_unsafe"] else 0.0
        cells[(m.group(1), int(m.group(2)))] = vec
    return cells


cells = load()
missing = [(r, s) for r in RATIOS for s in SEEDS if (r, s) not in cells]
if missing:
    raise SystemExit(f"missing cells: {missing}")
n_prompts = len(cells[("1to0", 42)])

# per-(seed, prompt) paired contrast
D = np.vstack([
    cells[("1to0", s)] - 0.5 * (cells[("2to1", s)] + cells[("1to1", s)])
    for s in SEEDS
])                                   # 3 x n_prompts
D_s = D.mean(axis=1)                 # block-level contrasts
D_bar = float(D_s.mean())

print(f"prompts = {n_prompts}, seed blocks = {len(SEEDS)}")
for s, d in zip(SEEDS, D_s):
    print(f"  seed {s}: block contrast D_s = {d:+.4f}")
print(f"D_bar (benign - mixed, paired within seed) = {D_bar:+.4f}\n")

# --- sign-flip procedure (assumes contrasts symmetric about zero under H0) ----
signs = list(itertools.product([-1, 1], repeat=len(SEEDS)))
stats = np.array([abs(float(np.mean(np.array(sg) * D_s))) for sg in signs])
p_signflip = float((stats >= abs(D_bar) - 1e-12).mean())
p_min = 2.0 / len(signs)
from scipy.stats import ttest_1samp
p_ttest = float(ttest_1samp(D_s, 0.0).pvalue)
print(f"sign-flip procedure over {len(signs)} sign assignments: p = {p_signflip:.4f}")
print(f"  smallest attainable two-sided p for THIS procedure = {p_min:.4f}")
print(f"paired t-test on the same contrasts (assumes normality): p = {p_ttest:.4f}")
print("  -> the two answers differ by the assumption, not by the data\n")

# --- prompt-level cluster bootstrap, seed blocks preserved --------------------
boot = np.empty(B)
for i in range(B):
    idx = RNG.integers(0, n_prompts, n_prompts)
    boot[i] = D[:, idx].mean()
ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
se = float(boot.std(ddof=1))
print(f"prompt cluster bootstrap (seeds fixed): SE = {se:.4f}, "
      f"95% CI = [{ci[0]:.4f}, {ci[1]:.4f}]")
print(f"  CI includes 0: {ci[0] <= 0 <= ci[1]}")
print("  (this generalizes over prompts only, conditional on the three realized seeds)")

out = {
    "design": ("complete matched block: 3 seed blocks x 3 ratio treatments on a "
               "common 200-prompt set; treatments assigned by construction, not "
               "randomized within blocks, so no analysis here is randomization-exact"),
    "judge": JUDGE, "n_prompts": n_prompts, "seeds": SEEDS,
    "block_contrasts": {str(s): round(float(d), 4) for s, d in zip(SEEDS, D_s)},
    "D_bar": round(D_bar, 4),
    "signflip_p_two_sided": round(p_signflip, 4),
    "signflip_min_attainable_p": round(p_min, 4),
    "signflip_assumption": "block contrasts independent and symmetric about zero under H0",
    "paired_ttest_p_two_sided": round(p_ttest, 4),
    "paired_ttest_assumption": "block contrasts normal; reported to show the floor is procedure-specific",
    "prompt_bootstrap_se": round(se, 4),
    "prompt_bootstrap_ci95": [round(ci[0], 4), round(ci[1], 4)],
    "prompt_bootstrap_scope": ("generalizes over prompts conditional on the three "
                               "realized seeds; not a seed-level inference"),
    "power_extrapolation": ("withdrawn: the crossed design's variance components are "
                            "not invariant additive pieces, so projecting to other "
                            "(n_prompt, n_seed) budgets from this run is unsupported"),
}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(out, open(OUT, "w", encoding="utf-8"), indent=2)
print(f"\nSaved {OUT}")
