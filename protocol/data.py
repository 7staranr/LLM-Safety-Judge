"""Shared data loading and configuration for the reproduction package.

Every analysis script imports the candidate-judge set, the prompt-source list,
the on-disk layout, and the label encoding from this module, so they are all
defined exactly once.

The released inputs are:

* ``human_annotation/<source>_adjudicated.json`` -- per-pair human labels.
  The four original judges' per-pair predictions are embedded as ``_judge_*``
  fields; ``source_ratio`` records the benign-to-safety SFT condition.
* ``analysis_results/local_judge_predictions/<judge>_<source>.json`` -- per-pair
  predictions for the five expansion judges (and for any embedded judge whose
  ``_judge_*`` field is absent on a given source, e.g. Llama Guard on
  ``harmful``).

Both files key every pair by ``annotation_id``.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Optional

# --- Candidate judges (K = 9), in paper order -------------------------------
JUDGES = [
    "qwen7b",
    "llama3b",
    "mistral7b",
    "llamaguard3_1b",
    "llama31_8b",
    "gemma2_9b",
    "phi3_medium",
    "wildguard",
    "shieldgemma_9b",
]

# Per-pair predictions for these four judges are embedded in the adjudicated
# JSON. The remaining five are read from the prediction files.
_EMBEDDED_FIELD = {
    "qwen7b": "_judge_qwen7b",
    "llama3b": "_judge_llama3b",
    "mistral7b": "_judge_mistral7b",
    "llamaguard3_1b": "_judge_llamaguard3",
}

# --- Prompt sources ---------------------------------------------------------
SOURCES = [
    "harmful",
    "sensitive",
    "harmbench",
    "xstest",
    "beavertails",
    "advbench",
    "multijail_zh",
]

# The five original English sources used for the main cross-source results.
ENGLISH_SOURCES = ["harmful", "sensitive", "harmbench", "xstest", "beavertails"]

# Short labels for compact console tables.
SOURCE_ABBR = {
    "harmful": "harm",
    "sensitive": "sens",
    "harmbench": "hbench",
    "xstest": "xstest",
    "beavertails": "beaver",
    "advbench": "advb",
    "multijail_zh": "mjail",
}

_ADJ_FILENAME = {
    "harmful": "natural_300_adjudicated.json",
    "sensitive": "sensitive_adjudicated.json",
    "harmbench": "harmbench_adjudicated.json",
    "xstest": "xstest_adjudicated.json",
    "beavertails": "beavertails_adjudicated.json",
    "advbench": "advbench_adjudicated.json",
    "multijail_zh": "multijail_zh_adjudicated.json",
}

# Benign-to-safety SFT ratios present in the calibration set.
RATIOS = ["1to0", "2to1", "1to1"]

# Label-aggregation rules supported by :func:`load_source` (used by the
# label-noise robustness analysis).
LABEL_RULES = ["adjudicated", "annotator_1", "annotator_2", "disagree_unsafe", "disagree_safe"]


def default_root() -> str:
    """Project root holding ``human_annotation/`` and ``analysis_results/``.

    The released package bundles these directories, so the default is the
    repository root, one level above ``protocol/``. Override with the
    ``DISTILL_ROOT`` environment variable to read them from elsewhere.
    """
    env = os.environ.get("DISTILL_ROOT")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, ".."))


def results_dir(root: Optional[str] = None) -> str:
    """Directory where analyses write their JSON outputs."""
    return os.path.join(root or default_root(), "analysis_results")


def adjudicated_path(source: str, root: Optional[str] = None) -> str:
    """Path to the human-adjudicated JSON for a prompt source."""
    return os.path.join(root or default_root(), "human_annotation", _ADJ_FILENAME[source])


def predictions_path(judge: str, source: str, root: Optional[str] = None) -> str:
    """Path to a judge's per-pair prediction file for a prompt source."""
    return os.path.join(
        root or default_root(), "analysis_results", "local_judge_predictions",
        f"{judge}_{source}.json",
    )


def _to_unsafe(value) -> Optional[bool]:
    """Map a SAFE/UNSAFE, boolean, or 0/1 label onto True (unsafe) or None."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().upper()
    if text in ("UNSAFE", "TRUE", "1"):
        return True
    if text in ("SAFE", "FALSE", "0"):
        return False
    return None


def _resolve_label(item: dict, rule: str) -> Optional[bool]:
    if rule == "adjudicated":
        return _to_unsafe(item.get("adjudicated"))
    ann1 = _to_unsafe(item.get("annotator_1"))
    ann2 = _to_unsafe(item.get("annotator_2"))
    if rule == "annotator_1":
        return ann1
    if rule == "annotator_2":
        return ann2
    if ann1 is None or ann2 is None:
        return None
    if ann1 == ann2:
        return ann1
    if rule == "disagree_unsafe":
        return True
    if rule == "disagree_safe":
        return False
    raise ValueError(f"unknown label_rule: {rule!r}")


def _load_predictions(root: str, judge: str, source: str) -> dict:
    path = predictions_path(judge, source, root)
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    return {
        pred["annotation_id"]: _to_unsafe(pred.get("label"))
        for pred in data.get("predictions", [])
    }


def load_source(
    source: str,
    root: Optional[str] = None,
    label_rule: str = "adjudicated",
) -> list:
    """Return per-pair records for one prompt source.

    Each record is a dict with keys:

    * ``annotation_id`` -- unique pair id.
    * ``true_unsafe`` -- human label under ``label_rule`` (True = unsafe, or None).
    * ``ratio`` -- the benign-to-safety SFT condition (``source_ratio``).
    * ``preds`` -- ``{judge: True/False/None}`` for all K = 9 judges.

    ``label_rule`` is one of :data:`LABEL_RULES`.
    """
    root = root or default_root()
    with open(adjudicated_path(source, root), encoding="utf-8") as handle:
        items = json.load(handle)

    pred_maps = {judge: _load_predictions(root, judge, source) for judge in JUDGES}

    records = []
    for item in items:
        preds = {}
        for judge in JUDGES:
            field = _EMBEDDED_FIELD.get(judge)
            value = _to_unsafe(item.get(field)) if field else None
            if value is None:
                value = pred_maps[judge].get(item["annotation_id"])
            preds[judge] = value
        records.append(
            {
                "annotation_id": item["annotation_id"],
                "true_unsafe": _resolve_label(item, label_rule),
                "ratio": item.get("source_ratio"),
                "preds": preds,
            }
        )

    # Surface silent candidate-set reduction: a judge with no predictions for
    # this source (missing prediction file, or absent _judge_* field with no
    # fallback file) is dropped downstream, which would change selection results.
    dropped = [j for j in JUDGES if all(r["preds"][j] is None for r in records)]
    if dropped:
        print(
            f"warning: source '{source}' has no predictions for {', '.join(dropped)}; "
            "these judges are dropped from the candidate set.",
            file=sys.stderr,
        )
    return records


def aligned_pairs(records: list, judge: str) -> tuple:
    """Return (human_unsafe, judge_unsafe) lists where both labels are present."""
    human, judged = [], []
    for record in records:
        truth = record["true_unsafe"]
        pred = record["preds"].get(judge)
        if truth is None or pred is None:
            continue
        human.append(truth)
        judged.append(pred)
    return human, judged


def calibrate(records: list, judges: Optional[list] = None, **metric_kwargs) -> dict:
    """Compute :class:`~protocol.judge_calibration.JudgeMetrics` per judge.

    Returns ``{judge: JudgeMetrics or None}``; the value is None when a judge
    has no overlapping labelled pairs on these records.
    """
    from protocol.judge_calibration import compute_metrics

    metrics = {}
    for judge in judges or JUDGES:
        human, judged = aligned_pairs(records, judge)
        metrics[judge] = compute_metrics(human, judged, **metric_kwargs) if human else None
    return metrics
