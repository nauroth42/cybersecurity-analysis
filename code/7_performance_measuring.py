"""
7_performance_measuring.py

Performance Measurement
=======================

Purpose
-------
Evaluates the best checkpoint of every model (selected by
6_checkpoint_selection.py) on the validation and the held-out test set,
and writes a model comparison table.

This is the only stage that touches the test set, so the reported test
figures are not used for any model or checkpoint choice.

Metrics (see ai_utils.multi_metrics)
------------------------------------
    accuracy (exact match), precision / recall / F1 (micro and macro),
    macro ROC-AUC, with a 0.50 probability threshold.

Input
-----
    CHECKPOINT_METRICS_CSV        from step 6
    <DATA>/val_1c.csv
    <DATA>/test_1c.csv

Output
------
    MODEL_COMPARISON_CSV
        One row per model: rank, model, checkpoint, selection score,
        val_* and test_* metrics. Ordered by the validation-based rank.

Configuration (config_ai.py)
----------------------------
    DATA, MODEL_DIR, CHECKPOINT_METRICS_CSV, MODEL_COMPARISON_CSV,
    EVAL_BATCH_SIZE

Author
------
Markus Nauroth, Southern Illinois University Edwardsville

Version
-------
1.1
"""

from __future__ import annotations

import gc
import tempfile
from pathlib import Path

import pandas as pd
import torch
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from ai_utils import load_tokenizer, multi_metrics, prepare_split
from config import (
    CHECKPOINT_METRICS_CSV,
    DATA,
    EVAL_BATCH_SIZE,
    MODEL_COMPARISON_CSV,
    MODEL_DIR,
)


def evaluate_checkpoint(ckpt_path: Path, val_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """Evaluate one saved checkpoint on the validation and test sets."""
    tok = load_tokenizer(ckpt_path)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_path)

    val_tok = prepare_split(val_df, tok)
    test_tok = prepare_split(test_df, tok)

    with tempfile.TemporaryDirectory() as tmp:
        args = TrainingArguments(
            output_dir=tmp,
            per_device_eval_batch_size=EVAL_BATCH_SIZE,
            report_to="none",
        )
        trainer = Trainer(model=model, args=args, compute_metrics=multi_metrics)
        val_metrics = trainer.evaluate(eval_dataset=val_tok, metric_key_prefix="val")
        test_metrics = trainer.evaluate(eval_dataset=test_tok, metric_key_prefix="test")

    del trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {**val_metrics, **test_metrics}


def main() -> None:
    best = pd.read_csv(CHECKPOINT_METRICS_CSV)
    if best.empty:
        raise SystemExit("Checkpoint table is empty; run 6_checkpoint_selection.py first.")

    data_dir = Path(DATA)
    val_df = pd.read_csv(data_dir / "val_1c.csv")
    test_df = pd.read_csv(data_dir / "test_1c.csv")

    rows = []
    for r in best.to_dict("records"):
        ckpt_path = Path(MODEL_DIR) / r["model"] / r["checkpoint"]
        print(f"\n{'=' * 60}\n  {r['model']}  ->  {r['checkpoint']}\n{'=' * 60}")

        metrics = evaluate_checkpoint(ckpt_path, val_df, test_df)
        row = {
            "rank": r.get("rank"),
            "model": r["model"],
            "checkpoint": r["checkpoint"],
            "selection_score": r.get("score"),
            **metrics,
        }
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("rank")
    Path(MODEL_COMPARISON_CSV).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(MODEL_COMPARISON_CSV, index=False)
    print(f"\nSaved comparison metrics -> {MODEL_COMPARISON_CSV}")
    print("Next: run 8_probability_extraction.py")


if __name__ == "__main__":
    main()
