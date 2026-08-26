# Auditable LLM Safety-Judge Selection via Confidence-Interval Filtering on Unsafe Recall

[![banner](paper-banner.png)](https://doi.org/10.1109/access.2026.3726251)

Reproduction code for the paper **Auditable LLM Safety-Judge Selection via
Confidence-Interval Filtering on Unsafe Recall**.

## Overview

Attack Success Rate (ASR) is widely used in large language model safety
evaluation, but the ASR reported by an automatic judge is not determined by the
evaluated model alone: it is also biased by the judge's false negatives, i.e.
unsafe responses that the judge labels as safe. This package implements the
paper's two-stage judge-screening protocol. It calibrates `K = 9` candidate
judges on human-labeled prompt-response data, admits only judges whose
Clopper-Pearson lower confidence bound on unsafe recall reaches a minimum
threshold, and selects among the admitted judges by balanced accuracy. When the
calibration evidence is insufficient, the protocol abstains instead of forcing a
judge selection.

The analyses cover cross-source calibration, the selection-rule ablations,
calibration-budget stress tests, label-noise robustness, the diagnostic
Rogan-Gladen ASR correction, and the cross-fitted MAE audit.

## Repository layout

```
reproduction/
|-- protocol/
|   |-- judge_calibration.py   metrics, Clopper-Pearson bounds, Stage 1/2 selection, Rogan-Gladen
|   `-- data.py                judge/source config + loaders for the released inputs
|-- analysis/                  one script per result (see below)
|   `-- revision/              peer-review revision analyses (see revision/README.md)
|-- runners/
|   `-- run_judge_local.py     GPU runner that produces per-pair judge predictions
|-- data/README.md             expected input files
|-- requirements.txt
`-- README.md
```

## Inputs

The scripts read the released human labels and judge predictions from the
project root (resolved automatically, or set the `DISTILL_ROOT` environment
variable / pass `--root`):

```
human_annotation/<source>_adjudicated.json          human labels (+ embedded predictions for 4 judges)
analysis_results/local_judge_predictions/<judge>_<source>.json   predictions for the 5 expansion judges
```

See `data/README.md` for the file list and field schema. The seven prompt
sources are `harmful`, `sensitive`, `harmbench`, `xstest`, `beavertails`,
`advbench`, `multijail_zh`; the `K = 9` judges are `qwen7b`, `llama3b`,
`mistral7b`, `llamaguard3_1b`, `llama31_8b`, `gemma2_9b`, `phi3_medium`,
`wildguard`, `shieldgemma_9b`.

## Analyses

Run any script with `-h` for its options (`--root`, `--out`, and per-analysis
flags such as `--bootstrap`).

| Script | Reproduces |
| --- | --- |
| `compute_k9_calibration.py` | Per-(source, judge) calibration metrics and the cross-source pass/reject pattern. Writes `k9_calibration.json`, the input for the ablations below. |
| `cross_source_protocol_ablation.py` | Selection-rule ablation R1-R5 per source: the metric-only rules can pick a Stage-1-failing judge, R5 abstains instead. |
| `ci_vs_point_estimate_ablation.py` | Confidence-interval filter vs. naive point-estimate filter; counts the over-confident admissions the CI rule prevents. |
| `calibration_size_stress.py` | Bootstrap stress over calibration-set size: Stage-1 pass frequency, selection stability, and abstention rate. |
| `calibration_size_stress_k4.py` | The same stress protocol restricted to the original K=4 pool, with Llama Guard 3-1B plotted explicitly (its aggregate counts match Qwen-7B's but 10/42 unsafe records differ, so the curves are not interchangeable). Source of the paper's stress-test figure. |
| `catastrophic_failure_rate.py` | Catastrophic-failure rate (selecting a full-data Stage-1-failing judge) per rule and calibration size. |
| `label_noise_robustness.py` | Stage-1 PASS/FAIL stability under five label-aggregation rules. |
| `cross_source_ablation_kfold.py` | 5-fold cross-fitted MAE of each rule against human-estimated ASR. |
| `corrected_asr.py` | Diagnostic Rogan-Gladen ASR correction with a percentile bootstrap CI. |

## Quick start (CPU)

```bash
pip install -r paper_ieee_access/reproduction/requirements.txt

cd paper_ieee_access/reproduction

# 1. Calibrate the K = 9 judges (writes analysis_results/k9_calibration.json)
python analysis/compute_k9_calibration.py

# 2. Ablations that consume k9_calibration.json
python analysis/cross_source_protocol_ablation.py
python analysis/ci_vs_point_estimate_ablation.py

# 3. Self-contained analyses (read the released inputs directly)
python analysis/label_noise_robustness.py
python analysis/cross_source_ablation_kfold.py
python analysis/calibration_size_stress.py --bootstrap 1000
python analysis/calibration_size_stress_k4.py --bootstrap 1000
python analysis/catastrophic_failure_rate.py --bootstrap 1000
python analysis/corrected_asr.py --bootstrap 1000
```

Each script prints a summary table and writes a JSON file to
`analysis_results/`.

## Regenerating judge predictions (GPU)

The released `local_judge_predictions/` files are produced by the GPU runner.
To regenerate them:

```bash
python runners/run_judge_local.py --judge llama31_8b --source all
```

Models load from their Hugging Face ids (bf16, or 4-bit NF4 for the 9B models;
see `--help`). Runs are resumable and only binary labels are persisted.

## Protocol API

```python
import sys; sys.path.insert(0, "paper_ieee_access/reproduction")
from protocol.judge_calibration import compute_metrics, select_primary_judge

human_unsafe = [True, True, False, False]
judge_outputs = {
    "judge_a": [True, False, False, False],
    "judge_b": [True, True, False, True],
}

metrics = {
    name: compute_metrics(human_unsafe, preds, tau_min=0.50)
    for name, preds in judge_outputs.items()
}
print(select_primary_judge(metrics))  # -> {'status': ..., 'selected': ..., ...}
```

## Acknowledgement

This work was supported in part by the National Natural Science Foundation of China under Grant 22374086, and in part by the National Key Research and Development Program of China under Grant 2023YFF0612100.

## Citation

```bibtex
@article{llmjudgeci, 
  title   = {Auditable LLM Safety-Judge Selection via Confidence-Interval Filtering on Unsafe Recall}, 
  author  = {Jixiang Yang, Junfei Yi, Jinhan Li and Shengjie Ma}, 
  journal = {IEEE Access}, 
  year    = {2026}
}
```
