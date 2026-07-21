"""Emit the design-weighted Stage-1 table (K=9) for the supplementary material."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "results", "design_weighted_stage1.json")
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "figures", "table_design_weighted.tex"))

NAMES = {"harmbench": "HarmBench", "xstest": "XSTest",
         "sensitive": "Sensitive (int.)", "beavertails": "BeaverTails",
         "harmful": "Harmful (int.)"}
JUDGE = {"qwen7b": "Qwen2.5-7B", "llama3b": "Llama-3.2-3B",
         "mistral7b": "Mistral-7B", "llamaguard3_1b": "Llama Guard 3-1B",
         "llama31_8b": "Llama-3.1-8B", "phi3_medium": "Phi-3-medium-4k",
         "gemma2_9b": "Gemma-2-9B", "shieldgemma_9b": "ShieldGemma-9B",
         "wildguard": "WildGuard"}
ORDER = ["harmbench", "xstest", "sensitive", "beavertails", "harmful"]
JORDER = ["qwen7b", "llama3b", "mistral7b", "llamaguard3_1b", "llama31_8b",
          "phi3_medium", "gemma2_9b", "shieldgemma_9b", "wildguard"]

d = json.load(open(SRC, encoding="utf-8"))
lines = []
for si, src in enumerate(ORDER):
    if src not in d["sources"]:
        continue
    judges = d["sources"][src]["judges"]
    first = True
    for j in JORDER:
        if j not in judges or judges[j].get("status") == "UNDERPOWERED":
            continue
        m = judges[j]
        label = NAMES[src] if first else ""
        mark = " $\\dagger$" if m["verdict_changed"] else ""
        lines.append(
            f"{label} & {JUDGE[j]} & {m['sens_naive']:.3f} & {m['lb_naive']:.3f} & "
            f"\\textsc{{{m['stage1_naive'].lower()}}} & {m['sens_weighted']:.3f} & "
            f"{m['lb_weighted']:.3f} & \\textsc{{{m['stage1_weighted'].lower()}}}{mark} \\\\"
        )
        first = False
    if si < len(ORDER) - 1:
        lines.append("\\midrule")

CAPTION = (
    "Unweighted versus design-weighted Stage~1 certification over the full "
    "$5 \\times 9$ English grid. Four of the five sources drew their labelled pairs "
    "with a stratified design keyed on judge outputs, targeting $50\\%$ judge-disagreement "
    "pairs, $30\\%$ unanimous-unsafe and $20\\%$ unanimous-safe subject to availability "
    "(the unanimous-unsafe stratum was exhausted as a census on every source, so its "
    "realized share is $2.8$--$18.6\\%$), so inclusion depends on the "
    "judge predictions and the unweighted sensitivity is not a natural-frame estimate. "
    "The design is known, so we undo it: strata are reconstructed from the judge "
    "predictions on each source pool, each pair is weighted by the inverse realized "
    "inclusion probability $N_h / n_h$, the point estimate is the Haj\\'{e}k "
    "design-weighted ratio of \\eqref{eq:hajek_sens}, and the bound comes from a Rao--Wu "
    "rescaled bootstrap ($B{=}4000$) carrying the finite-population correction, so census "
    "strata contribute no spurious variance. The internal harmful source is a "
    "natural-frame sample and acts as a control: there the weighted and unweighted "
    "point estimates are identical for all nine judges, which is what an inert "
    "reweighting should give. Their lower bounds still differ by up to $0.040$, but "
    "that residual reflects the interval rather than the weighting, since the "
    "unweighted column uses a Clopper--Pearson bound and the weighted column a "
    "bootstrap percentile. Of the $45$ cells, $44$ keep their verdict; the single "
    "flip ($\\dagger$) is Llama Guard 3-1B on XSTest, the same boundary cell that also "
    "flips under Bonferroni control and under one label-noise rule."
)

tex = (
    "\\begin{table*}[t]\n\\centering\n\\caption{" + CAPTION + "}\\label{tab:design_weighted}\n"
    "\\footnotesize\n\\setlength{\\tabcolsep}{5pt}\n"
    "\\begin{tabular}{llccclcc}\n\\toprule\n"
    "& & \\multicolumn{3}{c}{Unweighted} & \\multicolumn{3}{c}{Design-weighted} \\\\\n"
    "\\cmidrule(lr){3-5} \\cmidrule(lr){6-8}\n"
    "Source & Judge & Sens. & CI$_{\\mathrm{LB}}$ & Stage 1 & Sens. & CI$_{\\mathrm{LB}}$ & Stage 1 \\\\\n"
    "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n"
)

open(OUT, "w", encoding="utf-8").write(tex)
print("Wrote", OUT)
print(f"cells={d['n_cells']} flips={d['n_verdict_flips']}")

# 5/5 pass patterns under each estimator, for the main-text claim
for est in ("stage1_naive", "stage1_weighted"):
    full = []
    for j in JORDER:
        st = [d["sources"][s]["judges"].get(j, {}).get(est) for s in ORDER]
        if all(x == "PASS" for x in st):
            full.append(JUDGE[j])
    print(f"{est:18s} 5/5 judges: {full}")
