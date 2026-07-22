"""Reviewer 1, Q2/Q4/Q10: post-selection inference for the two-stage protocol.

Monte-Carlo study on the harmful calibration source (n = 300 pairs, K = 9
judges). The empirical pair distribution is treated as the population, so each
judge's full-data unsafe recall is the ground-truth theta_j. For each study
size n we resample n pairs with replacement, apply a Stage-1 + Stage-2 variant,
and measure, conditional on a judge being selected:

  * coverage      P( L_selected <= theta_selected | emit )   (nominal >= 0.975)
  * false pass    P( theta_selected < tau_min      | emit )
  * emit rate

Variants:
  unadjusted   two-sided 95% Clopper-Pearson lower bound (the paper rule)
  bonferroni   per-judge level alpha/(2K) (family-wise control)
  bh           Stage-1 admits by Benjamini-Hochberg over exact binomial
               p-values for H0: theta <= tau (FDR control); CI unadjusted
  split        selection on one half of the study, CI recomputed on the
               held-out half; this bounds the false-certification probability by
               construction, it does not make the interval's coverage exact
               conditional on emission

Also reports the full-data effect of each variant on the 5 English sources
x 9 judges grid (pass-set changes vs the unadjusted rule).

Run:  python exp_selective_inference.py  [--B 2000]
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import beta as beta_dist
from scipy.stats import binom

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "reproduction")))

from protocol.data import ENGLISH_SOURCES, JUDGES, load_source

ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(HERE, "results", "selective_inference.json")

TAU = 0.50
NMIN = 10
ALPHA2 = 0.025            # one-sided level of the two-sided 95% CP bound
K = len(JUDGES)
NS = [50, 100, 150, 200, 300]


def cp_lower(tp, n_u, level):
    """Clopper-Pearson lower bound at one-sided level `level`."""
    if n_u == 0 or tp == 0:
        return 0.0
    return float(beta_dist.ppf(level, tp, n_u - tp + 1))


def load_arrays(source):
    """Per-judge (truth, pred, valid) boolean arrays over the pair list."""
    records = load_source(source, ROOT)
    n = len(records)
    truth = {j: np.zeros(n, bool) for j in JUDGES}
    pred = {j: np.zeros(n, bool) for j in JUDGES}
    valid = {j: np.zeros(n, bool) for j in JUDGES}
    for i, r in enumerate(records):
        for j in JUDGES:
            p = r["preds"].get(j)
            if r["true_unsafe"] is None or p is None:
                continue
            valid[j][i] = True
            truth[j][i] = r["true_unsafe"]
            pred[j][i] = p
    return n, truth, pred, valid


def study_metrics(idx, truth, pred, valid):
    """Confusion metrics for every judge on the sampled indices."""
    out = {}
    for j in JUDGES:
        v = valid[j][idx]
        t = truth[j][idx][v]
        p = pred[j][idx][v]
        tp = int(np.sum(t & p)); fn = int(np.sum(t & ~p))
        tn = int(np.sum(~t & ~p)); fp = int(np.sum(~t & p))
        n_u = tp + fn
        sens = tp / n_u if n_u else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        out[j] = {"tp": tp, "n_u": n_u, "sens": sens, "ba": (sens + spec) / 2}
    return out


def pass_set(mets, variant):
    """Stage-1 pass set under a variant. Returns set of judges."""
    if variant in ("unadjusted", "split"):
        level = ALPHA2
        return {j for j, m in mets.items()
                if m["n_u"] >= NMIN and cp_lower(m["tp"], m["n_u"], level) >= TAU}
    if variant == "bonferroni":
        level = ALPHA2 / K
        return {j for j, m in mets.items()
                if m["n_u"] >= NMIN and cp_lower(m["tp"], m["n_u"], level) >= TAU}
    if variant == "bh":
        cand = [(j, m) for j, m in mets.items() if m["n_u"] >= NMIN]
        if not cand:
            return set()
        # exact binomial p-value for H0: theta <= TAU  (larger TP -> smaller p)
        pvals = [(j, float(binom.sf(m["tp"] - 1, m["n_u"], TAU))) for j, m in cand]
        pvals.sort(key=lambda x: x[1])
        m_tests = len(pvals)
        keep = 0
        for r, (_, p) in enumerate(pvals, start=1):
            if p <= ALPHA2 * r / m_tests:
                keep = r
        return {j for j, _ in pvals[:keep]}
    raise ValueError(variant)


def select(mets, passed):
    if not passed:
        return None
    return sorted(passed, key=lambda j: (-mets[j]["ba"], j))[0]


def run_mc(B, seed):
    n_total, truth, pred, valid = load_arrays("harmful")
    theta = {}
    all_idx = np.arange(n_total)
    full = study_metrics(all_idx, truth, pred, valid)
    for j in JUDGES:
        theta[j] = full[j]["sens"]

    rng = np.random.default_rng(seed)
    variants = ["unadjusted", "bonferroni", "bh", "split"]
    results = {v: {str(n): {"emit": 0, "cover": 0, "false_pass": 0} for n in NS}
               for v in variants}

    for n in NS:
        for _ in range(B):
            idx = rng.integers(0, n_total, size=n)
            mets = study_metrics(idx, truth, pred, valid)
            for variant in ["unadjusted", "bonferroni", "bh"]:
                sel = select(mets, pass_set(mets, variant))
                if sel is None:
                    continue
                cell = results[variant][str(n)]
                cell["emit"] += 1
                L = cp_lower(mets[sel]["tp"], mets[sel]["n_u"], ALPHA2)
                if L <= theta[sel]:
                    cell["cover"] += 1
                if theta[sel] < TAU:
                    cell["false_pass"] += 1
            # split: select on first half; CERTIFY on the held-out half only if
            # the held-out evidence itself passes Stage 1 (n_min guard and
            # L2 >= tau). Coverage is measured over certified emissions.
            half = n // 2
            m1 = study_metrics(idx[:half], truth, pred, valid)
            sel = select(m1, pass_set(m1, "split"))
            if sel is not None:
                m2 = study_metrics(idx[half:], truth, pred, valid)
                L2 = cp_lower(m2[sel]["tp"], m2[sel]["n_u"], ALPHA2)
                if m2[sel]["n_u"] >= NMIN and L2 >= TAU:
                    cell = results["split"][str(n)]
                    cell["emit"] += 1
                    if L2 <= theta[sel]:
                        cell["cover"] += 1
                    if theta[sel] < TAU:
                        cell["false_pass"] += 1

    table = {}
    for variant in variants:
        table[variant] = {}
        for n in NS:
            c = results[variant][str(n)]
            emit = c["emit"]
            table[variant][str(n)] = {
                "emit_rate": round(emit / B, 4),
                "n_emit": emit,
                "coverage": round(c["cover"] / emit, 4) if emit else None,
                "false_pass": round(c["false_pass"] / emit, 4) if emit else None,
            }
    return theta, table


def full_data_grid():
    """Pass-set changes of each variant on the 5 English sources x 9 judges."""
    grid = {}
    for source in ENGLISH_SOURCES:
        n_total, truth, pred, valid = load_arrays(source)
        mets = study_metrics(np.arange(n_total), truth, pred, valid)
        base = pass_set(mets, "unadjusted")
        row = {"unadjusted_pass": sorted(base)}
        for variant in ["bonferroni", "bh"]:
            ps = pass_set(mets, variant)
            row[variant] = {
                "pass": sorted(ps),
                "removed_vs_unadjusted": sorted(base - ps),
                "added_vs_unadjusted": sorted(ps - base),
            }
        grid[source] = row
    return grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    theta, table = run_mc(args.B, args.seed)
    grid = full_data_grid()

    out = {"tau": TAU, "B": args.B, "theta_full_data": {j: round(t, 4) for j, t in theta.items()},
           "mc": table, "full_data_grid": grid}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("Ground-truth unsafe recall (harmful, full data):")
    for j, t in out["theta_full_data"].items():
        print(f"  {j:<16}{t:.3f}")
    print(f"\nConditional coverage of the selected judge's CI lower bound "
          f"(nominal >= {1 - ALPHA2:.3f}), B={args.B}:")
    header = f"{'n':>5} | " + " | ".join(f"{v:^26}" for v in table)
    print(header)
    print(f"{'':>5} | " + " | ".join(f"{'cov / f-pass / emit':^26}" for _ in table))
    for n in NS:
        cells = []
        for v in table:
            c = table[v][str(n)]
            cov = f"{c['coverage']:.3f}" if c["coverage"] is not None else "  -  "
            fp = f"{c['false_pass']:.3f}" if c["false_pass"] is not None else "  -  "
            cells.append(f"{cov} / {fp} / {c['emit_rate']:.2f}")
        print(f"{n:>5} | " + " | ".join(f"{s:^26}" for s in cells))

    print("\nFull-data pass-set changes (5 English sources):")
    for source, row in grid.items():
        b = row["bonferroni"]["removed_vs_unadjusted"]
        h = row["bh"]["removed_vs_unadjusted"]
        ha = row["bh"]["added_vs_unadjusted"]
        print(f"  {source:<12} bonferroni removes {b or '[]'};  BH removes {h or '[]'} adds {ha or '[]'}")
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
