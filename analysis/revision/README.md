# Revision-round analyses

These scripts implement the analyses added during peer review. They accompany the
supplementary material's revision section and the response to reviewers. Each
reads the released inputs through the loaders in `../../protocol/` (which resolve
`../../human_annotation/` and `../../analysis_results/` once `DISTILL_ROOT` points
at the repository root; see `../../data/README.md`), or the aggregated result files
in `results/`, and writes the numbers and LaTeX tables the paper cites. None require
a GPU; all are CPU re-analyses of the released predictions.

| Script | Produces | Paper location |
| --- | --- | --- |
| `exp_selective_inference.py` | Monte-Carlo post-selection coverage, Bonferroni/BH/split variants | main-text coverage table |
| `make_coverage_ci_table.py` | exact Clopper-Pearson intervals on each coverage cell | supplementary coverage-CI table |
| `exp_design_weighted_stage1.py` | Hajek design-weighted Stage-1 bounds with a Rao-Wu rescaled bootstrap (K=9) | supplementary design-weighted table |
| `make_design_weighted_table.py` | LaTeX for the design-weighted table | supplementary |
| `exp_spec_tiebreak.py` | specificity/sensitivity Stage-2 tie-break MAE, signed bias, FPR, false-positive counts | supplementary tie-break table |
| `exp_boundary_samplesize.py` | persistent boundary sample-size thresholds | supplementary boundary table |
| `make_sample_size_table.py` | LaTeX for the theta x tau sample-size grid | supplementary |
| `exp_phi_paired.py` | Phi matched-block contrast, sign-flip and paired-t p-values, prompt bootstrap | main-text Phi subsection |
| `exp_common_denominator.py` | common-denominator deployment audit; separates judge-to-judge spread from the underestimation factor | main-text deployment-audit paragraph |
| `exp_decomposition_consistent.py` | filter/correction decomposition on one consistent basis | main-text decomposition |
| `exp_parse_convention.py` | non-parse rates and both parse conventions' Stage-1 verdicts | supplementary parsing note |
| `exp_tau_sensitivity.py` | tau_min sweep over the K=9 x 7-source grid | main-text threshold sensitivity |
| `analyze_multilingual.py` | MultiJail-zh remedies (zh prompt, ShieldLM, Qwen3Guard) | supplementary multilingual table |
| `rebuild_kappa_table.py` | inter-annotator agreement recomputed from the adjudicated labels | supplementary agreement table |

`results/` holds the aggregated JSON outputs of these scripts, so the tables can
be regenerated without re-running the full pipeline.
