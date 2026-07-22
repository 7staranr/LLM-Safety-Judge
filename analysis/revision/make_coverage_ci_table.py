"""Coverage point estimates with Monte-Carlo uncertainty (re-review section 5, item 3).

The main-text coverage table reports point estimates to three decimals. The
reviewer correctly notes that some rest on few emissions (the split arm at n=100
is 33/33 = 1.000), so a point estimate alone overstates the evidence. This emits
a supplementary table giving, for every (variant, n) cell, the coverage as a
numerator/denominator fraction with a Clopper-Pearson 95% interval and the
Monte-Carlo standard error, computed from the same B=2000 replicate counts.
"""
import json
import os

from scipy.stats import beta as beta_dist

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "results", "selective_inference.json")
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "figures", "table_coverage_ci.tex"))

VAR = [("unadjusted", "Unadjusted"), ("bonferroni", "Bonferroni"),
       ("bh", "BH"), ("split", "Split")]
NS = ["50", "100", "150", "200", "300"]


def cp(k, n, a=0.05):
    lo = 0.0 if k == 0 else float(beta_dist.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta_dist.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


d = json.load(open(SRC, encoding="utf-8"))["mc"]
rows = []
print(f"{'variant':<11}{'n':>5}{'cov':>8}{'num/den':>10}{'CP 95% CI':>20}{'MC SE':>9}")
for key, label in VAR:
    for n in NS:
        cell = d[key].get(n, {})
        emit = cell.get("n_emit", 0)
        covp = cell.get("coverage")
        if not emit or covp is None:
            rows.append((label, n, None))
            print(f"{label:<11}{n:>5}{'--':>8}{'0/0':>10}{'--':>20}{'--':>9}")
            continue
        num = round(covp * emit)
        lo, hi = cp(num, emit)
        se = (covp * (1 - covp) / emit) ** 0.5
        rows.append((label, n, (covp, num, emit, lo, hi, se)))
        print(f"{label:<11}{n:>5}{covp:>8.3f}{f'{num}/{emit}':>10}"
              f"{f'[{lo:.3f}, {hi:.3f}]':>20}{se:>9.4f}")

body = []
for label, n, v in rows:
    if v is None:
        body.append(f"{label} & {n} & --- & --- & --- \\\\")
    else:
        covp, num, emit, lo, hi, se = v
        body.append(f"{label} & {n} & {num}/{emit} & {covp:.3f} & "
                    f"$[{lo:.3f}, {hi:.3f}]$ \\\\")
# group by variant with midrules
grouped = []
last = None
for line in body:
    lab = line.split(" & ")[0]
    if last is not None and lab != last:
        grouped.append("\\midrule")
    grouped.append(line)
    last = lab

CAP = ("Monte-Carlo conditional-coverage estimates of the main-text coverage table with their "
       "sampling uncertainty. Coverage is shown as the number of covered emissions over the "
       "number emitted, with a Clopper--Pearson $95\\%$ interval. The point estimates are tight "
       "where many replicates emit (the unadjusted rule at $n\\geq 100$ rests on $1{,}700$ or "
       "more emissions) and weak where few do: the split arm at $n=100$ is $33/33$, so its "
       "$1.000$ estimate carries a one-sided interval reaching down to $0.894$ and should be "
       "read as consistent-with-conservative rather than as established exact coverage. We use "
       "the exact Clopper--Pearson interval rather than a normal-approximation standard error, "
       "which degenerates to zero at boundary cells such as the $33/33$ split arm.")

tex = ("\\begin{table}[!t]\n\\centering\n\\caption{" + CAP + "}\\label{tab:coverage_ci}\n"
       "\\footnotesize\n\\begin{tabular}{llccc}\n\\toprule\n"
       "Variant & $n$ & covered/emitted & coverage & CP $95\\%$ CI \\\\\n\\midrule\n"
       + "\n".join(grouped) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
open(OUT, "w", encoding="utf-8").write(tex)
print(f"\nWrote {OUT}")
