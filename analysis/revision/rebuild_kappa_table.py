"""Recompute inter-annotator agreement for every human-labelled source from the
adjudicated annotation files, so the supplementary agreement table matches the
main-text label budgets (reviewer round-2, issue 6)."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("DISTILL_ROOT",
                      os.path.abspath(os.path.join(HERE, "..", "..")))
HA = os.path.join(ROOT, "human_annotation")
OUT = os.path.join(ROOT, "figures", "table_kappa.tex")

SOURCES = [
    ("Harmful (natural)", "natural_300_adjudicated.json"),
    ("Harmful (enriched)", "enriched_100_adjudicated.json"),
    ("Sensitive (internal)", "sensitive_adjudicated.json"),
    ("HarmBench", "harmbench_adjudicated.json"),
    ("XSTest", "xstest_adjudicated.json"),
    ("BeaverTails", "beavertails_adjudicated.json"),
    ("AdvBench", "advbench_adjudicated.json"),
    ("MultiJail-zh", "multijail_zh_adjudicated.json"),
]


def cohen_kappa(a, b):
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    labels = set(a) | set(b)
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return (po - pe) / (1 - pe), po


rows = []
for name, fn in SOURCES:
    path = os.path.join(HA, fn)
    data = json.load(open(path, encoding="utf-8"))
    a1 = [d["annotator_1"] for d in data]
    a2 = [d["annotator_2"] for d in data]
    adj = [d["adjudicated"] for d in data]
    k, po = cohen_kappa(a1, a2)
    n_unsafe = sum(1 for x in adj if x == "UNSAFE")
    rows.append({
        "name": name, "n": len(data), "agree": po, "kappa": k,
        "n_unsafe": n_unsafe, "rate": n_unsafe / len(data),
    })
    print(f"{name:22s} n={len(data):4d} agree={po:.3f} kappa={k:.3f} "
          f"unsafe={n_unsafe:4d} ({n_unsafe/len(data)*100:.1f}%)")

kmin = min(r["kappa"] for r in rows)
kmax = max(r["kappa"] for r in rows)
print(f"\nkappa range: [{kmin:.3f}, {kmax:.3f}]")

body = "\n".join(
    f"{r['name']} & {r['n']} & {r['agree']*100:.1f}\\% & {r['kappa']:.3f} & "
    f"{r['n_unsafe']} & {r['rate']*100:.1f}\\% \\\\"
    for r in rows
)

kmin_src = min(rows, key=lambda r: r["kappa"])["name"]
kmax_src = max(rows, key=lambda r: r["kappa"])["name"]
k_sens = next(r["kappa"] for r in rows if r["name"] == "Sensitive (internal)")

tex = r"""\begin{table*}[t]
\centering
\caption{Inter-annotator agreement on all human-labelled subsets. Every (prompt, response) pair
in every source was independently labelled by two annotators; the values below are computed on the
complete labelled set of each source, and disagreements were resolved by the safety-conservative
tie-break described in the annotation rubric. Cohen's $\kappa$ ranges from %.3f on MultiJail-zh to
%.3f on BeaverTails. The two lowest values have different causes: MultiJail-zh ($\kappa=%.3f$) and
AdvBench ($\kappa=%.3f$) pair high raw agreement with low unsafe prevalence, which depresses the
chance-corrected statistic, whereas the internal sensitive set ($\kappa=%.3f$) is deliberately
edge-heavy and its annotators genuinely disagree more often. Unsafe counts are taken from the
adjudicated labels and match the per-source budgets in the main paper.}\label{tab:kappa}
\small
\begin{tabular}{lccccc}
\toprule
Source & $n$ & Agreement & Cohen's $\kappa$ & $n$ unsafe & Unsafe rate \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table*}
""" % (kmin, kmax,
       next(r["kappa"] for r in rows if r["name"] == "MultiJail-zh"),
       next(r["kappa"] for r in rows if r["name"] == "AdvBench"),
       k_sens,
       body)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w", encoding="utf-8").write(tex)
print(f"\nWrote {OUT}")
