"""Boundary-judge sample-size planning (Reviewer 1, Q6).

For a judge with true unsafe recall theta, compute the minimum number of unsafe
positives n_u such that the Stage-1 verdict is stable, i.e.

    n_min(theta, tau) = min { n : P[ L(K, n) >= tau ] >= target },  K ~ Bin(n, theta)

with L the Clopper-Pearson lower bound at alpha = 0.05. Reported for targets
0.80 / 0.90, plus the implied total label budget at the harmful source's unsafe
prevalence.

Note on theta <= tau: the pass probability then cannot reach the target at any n,
because a judge sitting at the floor clears the bound at most about half the time
regardless of sample size. Such cells are reported as N/A, distinct from cells
that merely exceed the n <= 2000 search range.

This was previously Part A of exp_samplesize_power.py, whose Part B (the Phi seed
power projection) was withdrawn during review; see deprecated/README.md. Part A is
unaffected by that withdrawal and is split out here so no live claim depends on a
deprecated script.

Run: python exp_boundary_samplesize.py
"""
import json
import math
import os

from scipy.stats import beta as beta_dist
from scipy.stats import binom

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "boundary_samplesize.json")

TAU = 0.50
ALPHA2 = 0.025
PREVALENCE = 42 / 300      # unsafe prevalence on the internal harmful source


def cp_lower(tp, n_u):
    if n_u == 0 or tp == 0:
        return 0.0
    return float(beta_dist.ppf(ALPHA2, tp, n_u - tp + 1))


def pass_probability(theta, n_u):
    """P[ CP lower bound >= TAU ] when TP ~ Bin(n_u, theta)."""
    lo, hi = 0, n_u
    while lo < hi:
        mid = (lo + hi) // 2
        if cp_lower(mid, n_u) >= TAU:
            hi = mid
        else:
            lo = mid + 1
    if cp_lower(lo, n_u) < TAU:
        return 0.0
    return float(binom.sf(lo - 1, n_u, theta))


NMAX = 2000


def persistent_threshold(theta, target):
    """Smallest n such that the target holds for EVERY n' in [n, NMAX].

    The pass probability is not monotone in n: the Clopper-Pearson bound is a step
    function of the discrete count, so the probability dips as the critical count
    ticks over. A first crossing is therefore not a requirement one can rely on
    (for theta=0.65 at tau=0.50 the target is first met at n=61, but n=91 falls
    back to 0.79). We scan downward from NMAX and return the point past which the
    target never fails again.
    """
    probs = {n: pass_probability(theta, n) for n in range(10, NMAX + 1)}
    n_star = None
    for n in range(NMAX, 9, -1):
        if probs[n] >= target:
            n_star = n
        else:
            break
    return n_star, probs


rows = []
print(f"Persistent minimum unsafe positives for a stable PASS at tau={TAU}")
print("(smallest n beyond which the target holds for every larger n <= 2000)")
print(f"  {'theta':>6}{'n_u (80%)':>12}{'labels (80%)':>14}{'n_u (90%)':>12}{'labels (90%)':>14}")
for theta in [0.55, 0.60, 0.65, 0.70, 0.80]:
    row = {"theta": theta}
    for tgt in [0.80, 0.90]:
        n_min, probs = persistent_threshold(theta, tgt)
        first = next((n for n in range(10, NMAX + 1) if probs[n] >= tgt), None)
        row[f"n_unsafe_p{int(tgt*100)}"] = n_min
        row[f"first_crossing_p{int(tgt*100)}"] = first
        row[f"total_labels_p{int(tgt*100)}"] = (
            math.ceil(n_min / PREVALENCE) if n_min else None)
    rows.append(row)
    print(f"  {theta:>6.2f}{str(row['n_unsafe_p80']):>12}{str(row['total_labels_p80']):>14}"
          f"{str(row['n_unsafe_p90']):>12}{str(row['total_labels_p90']):>14}"
          f"   (first crossings: {row['first_crossing_p80']}, {row['first_crossing_p90']})")
print(f"  (total labels assume harmful-source unsafe prevalence {PREVALENCE:.3f})")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump({"tau": TAU, "alpha_two_sided": 2 * ALPHA2,
           "prevalence_harmful": round(PREVALENCE, 4),
           "search_range": [10, NMAX],
           "threshold_type": ("persistent: smallest n beyond which the target holds "
                              "for every larger n in the search range; first crossings "
                              "are reported alongside because the pass probability is "
                              "non-monotone in n"),
           "rows": rows},
          open(OUT, "w", encoding="utf-8"), indent=2)
print(f"\nSaved {OUT}")

# Emit the supplementary table from the same numbers, so it cannot drift from them.
TEX = os.path.abspath(os.path.join(HERE, "..", "..", "figures", "table_boundary_n.tex"))
body = "\n".join(
    f"{r['theta']:.2f} & {r['n_unsafe_p80']} & {r['total_labels_p80']:,} & "
    f"{r['n_unsafe_p90']} & {r['total_labels_p90']:,} \\\\".replace(",", "{,}")
    for r in rows)
cap = ("Persistent minimum unsafe positives $n_{\\mathrm{u}}$ (and implied total labels at the "
       "harmful-source unsafe prevalence of $0.14$) for a stable Stage-1 PASS at "
       "$\\tau_{\\min}=0.50$, as a function of the judge's true unsafe recall $\\theta$. "
       "Because the pass probability is not monotone in $n$, these are the smallest sample "
       "sizes beyond which the target holds for \\emph{every} larger $n$ in the search range "
       "($n \\leq 2{,}000$), not the first $n$ at which it is met. First crossings sit "
       "$5$--$10\\%$ lower (for example $90$ rather than $97$ at $\\theta=0.65$) and are not "
       "reliable planning figures, because the probability can fall back below the target at "
       "the next few sample sizes.")
os.makedirs(os.path.dirname(TEX), exist_ok=True)
open(TEX, "w", encoding="utf-8").write(
    "\\begin{table}[!t]\n\\centering\n\\caption{" + cap + "}\\label{tab:boundary_n}\n"
    "\\footnotesize\n\\begin{tabular}{lcccc}\n\\toprule\n"
    "$\\theta$ & $n_{\\mathrm{u}}$ (80\\%) & labels (80\\%) & $n_{\\mathrm{u}}$ (90\\%) "
    "& labels (90\\%) \\\\\n\\midrule\n" + body
    + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
print(f"Wrote {TEX}")
