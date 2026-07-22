# Data Notes

The inputs the analysis scripts read are bundled in this repository so it is
self-contained. Point `DISTILL_ROOT` at the repository root before running any
script (the loaders otherwise default to a path outside the package):

```bash
export DISTILL_ROOT="$(pwd)"        # from the repository root
python analysis/compute_k9_calibration.py
```

The `<source>` prompts come from the paper's internal harmful and sensitive sets
(released here) and from public benchmarks (HarmBench, XSTest, BeaverTails,
AdvBench, MultiJail); the benchmark-derived prompts are redistributed for
reproduction under their original licenses, which should be consulted for reuse.
The `response` fields are the paper's own SFT-student generations, and the judge
predictions are the paper's own model outputs.

Human-adjudicated files (all included under `human_annotation/`):

```text
human_annotation/natural_300_adjudicated.json
human_annotation/sensitive_adjudicated.json
human_annotation/harmbench_adjudicated.json
human_annotation/xstest_adjudicated.json
human_annotation/beavertails_adjudicated.json
human_annotation/advbench_adjudicated.json
human_annotation/multijail_zh_adjudicated.json
```

Each adjudicated JSON is a list of pairs. Each pair carries the human label in
`adjudicated` (`"SAFE"` / `"UNSAFE"`), the two independent annotations
(`annotator_1`, `annotator_2`, used by the label-noise analysis), the SFT
condition (`source_ratio`), and embedded boolean predictions for the four
original judges (`_judge_*`, where `true` means unsafe):

```json
{
  "annotation_id": "harmful_0001",
  "prompt": "...",
  "response": "...",
  "adjudicated": "UNSAFE",
  "annotator_1": "UNSAFE",
  "annotator_2": "SAFE",
  "source_ratio": "1to0",
  "_judge_qwen7b": true,
  "_judge_llama3b": true,
  "_judge_mistral7b": false,
  "_judge_llamaguard3": true
}
```

Labels are decoded by `protocol.data` from `"SAFE"`/`"UNSAFE"`, booleans, or
`0`/`1`. An embedded `_judge_*` field may be absent on a given source (e.g.
`_judge_llamaguard3` on `harmful`), in which case that judge's predictions are
read from its prediction file instead.

Predictions for the five expansion judges (and for any embedded judge missing
its `_judge_*` field) are stored separately as `{"annotation_id": ..., "label":
"SAFE"|"UNSAFE"}` entries:

```text
analysis_results/local_judge_predictions/{judge}_{source}.json
```

where `judge` is one of `llama31_8b`, `gemma2_9b`, `phi3_medium`, `wildguard`,
`shieldgemma_9b` (or one of the four originals as a fallback), and `source` is
one of the seven prompt-source keys used in the paper.
