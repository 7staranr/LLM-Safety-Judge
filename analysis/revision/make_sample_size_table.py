"""Regenerate the supplementary sample-size table with persistent thresholds.

Reviewer round-5, issue 4: the published table reported first crossings of the
0.80 target, but the pass probability is not monotone in n (the Clopper-Pearson
bound is a step function of the discrete count), so a first crossing is not a
requirement a planner can rely on. This recomputes the same (theta, tau) grid as
persistent thresholds: the smallest n beyond which the target holds for every
larger n in the search range.
"""
import json
import os

from scipy.stats import beta as beta_dist
from scipy.stats import binom

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "figures", "table_sample_size.tex"))
JOUT = os.path.join(HERE, "results", "sample_size_grid.json")

ALPHA2 = 0.025
TARGET = 0.80
NMAX = 2000
THETAS = [0.65, 0.75, 0.85, 0.95]
TAUS = [0.50, 0.70, 0.85, 0.90, 0.95]


def cp_lower(tp, n):
    return 0.0 if (n == 0 or tp == 0) else float(beta_dist.ppf(ALPHA2, tp, n - tp + 1))


def pass_prob(theta, n, tau):
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        if cp_lower(mid, n) >= tau:
            hi = mid
        else:
            lo = mid + 1
    return 0.0 if cp_lower(lo, n) < tau else float(binom.sf(lo - 1, n, theta))


def persistent(theta, tau):
    probs = [pass_prob(theta, n, tau) for n in range(10, NMAX + 1)]
    n_star = None
    for i in range(len(probs) - 1, -1, -1):
        if probs[i] >= TARGET:
            n_star = i + 10
        else:
            break
    return n_star


grid, lines = {}, []
print(f"Persistent n_min for pass probability >= {TARGET} (search n <= {NMAX})")
header = "True $\\theta$ \\textbackslash\\ $\\tau_{\\min}$ & " + " & ".join(
    f"$\\tau{{=}}{t:.2f}$" for t in TAUS) + " \\\\"
for th in THETAS:
    cells, row = [], {}
    for ta in TAUS:
        n = persistent(th, ta) if th > ta else None
        row[str(ta)] = n
        cells.append(f"${n}$" if n else "N/A")
    grid[str(th)] = row
    lines.append(f"$\\theta={th:.2f}$ & " + " & ".join(cells) + " \\\\")
    print(f"  theta={th:.2f}: " + "  ".join(f"tau={ta}: {row[str(ta)]}" for ta in TAUS))

CAP = ("Persistent minimum unsafe-positive labels $n^{\\min}_{\\mathrm{unsafe}}$ for a judge with "
       "true unsafe recall $\\theta$ to pass Stage~1 with probability $\\geq 0.80$ at floor "
       "$\\tau_{\\min}$ (numerically evaluated definition in the cross-source generalization "
       "subsection of the main paper). The pass probability is not monotone in $n$, because the "
       "Clopper--Pearson bound is a step function of the discrete unsafe-positive count, so we "
       "report the smallest $n$ beyond which the target holds for \\emph{every} larger $n$ in "
       "the search range rather than the first $n$ at which it is met; first crossings are "
       "$5$--$10\\%$ lower and are not reliable planning figures. ``N/A'' = the target is not "
       "reached within $n \\leq 2{,}000$. For $\\theta \\leq \\tau$ it is unreachable at any $n$: "
       "the two-sided $95\\%$ bound satisfies $\\Pr[L > \\theta] \\leq 0.025$ by construction, so a "
       "judge sitting at the floor clears it at most about $2.5\\%$ of the time however much "
       "data it has. Certifying a judge for $\\tau \\geq 0.85$ therefore requires either "
       "$\\theta \\geq 0.95$ or $n_{\\mathrm{unsafe}} \\gg 100$.")

tex = ("\\begin{table*}[h]\n\\centering\n\\caption{" + CAP + "}\\label{tab:sample_size}\n"
       "\\small\n\\begin{tabular}{lccccc}\n\\toprule\n" + header + "\n\\midrule\n"
       + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
open(OUT, "w", encoding="utf-8").write(tex)
os.makedirs(os.path.dirname(JOUT), exist_ok=True)
json.dump({"target": TARGET, "alpha_two_sided": 2 * ALPHA2, "search_max_n": NMAX,
           "threshold_type": "persistent", "grid": grid},
          open(JOUT, "w", encoding="utf-8"), indent=2)
print(f"\nWrote {OUT}\nWrote {JOUT}")
