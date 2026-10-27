"""
5_transformer_finetuning.py

Transformer Fine-Tuning
=======================

Purpose
-------
Fine-tunes each transformer listed in MODELS on the multi-label
classification task (e.g. SEC Item 1C cybersecurity disclosures) and
saves its checkpoints. This script ONLY trains; the later stages are:

    6_checkpoint_selection.py   picks the best checkpoint per model
    7_performance_measuring.py  evaluates those checkpoints on val/test
    8_probability_extraction.py scores unseen documents

Pipeline Overview
-----------------
    1_data_acquisition.py
    2_data_extraction_cleansing.py
    3_data_labeling.py
    4_data_partitioning.py
    5_transformer_finetuning.py      <- this script
    6_checkpoint_selection.py
    7_performance_measuring.py
    8_probability_extraction.py

Input Files
-----------
    <DATA>/train_1c.csv
    <DATA>/val_1c.csv     (used for per-epoch validation during training)

Each needs a 'text' column plus the label columns in LABEL_COLS.
The test set is deliberately NOT touched here; it is used only in step 7.

Output
------
One folder per model under MODEL_DIR, e.g.

    models/bert-base-uncased/
        checkpoint-<step>/          model, tokenizer, trainer_state.json
        config.json, model files    best model (load_best_model_at_end)
        tokenizer files

Configuration (config_ai.py)
----------------------------
    DATA, MODEL_DIR, LABEL_COLS, NUM_EPOCHS, TRAIN_BATCH_SIZE,
    EVAL_BATCH_SIZE, LOGGING_STEPS, METRIC_KEY, SAVE_TOTAL_LIMIT

METRIC_KEY must be one of the keys returned by ai_utils.multi_metrics,
e.g. "f1_macro" or "f1_micro".

Dependencies
------------
pandas, torch, datasets, transformers, accelerate, scikit-learn,
sentencepiece and protobuf (for DeBERTa-v3)

Author
------
Markus Nauroth, Southern Illinois University Edwardsville

Version
-------
1.1
"""

from __future__ import annotations

import gc
from pathlib import Path

import pandas as pd
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from ai_utils import multi_metrics, prepare_split, safe_name
from config import (
    DATA,
    EVAL_BATCH_SIZE,
    LABEL_COLS,
    LOGGING_STEPS,
    METRIC_KEY,
    MODEL_DIR,
    NUM_EPOCHS,
    SAVE_TOTAL_LIMIT,
    TRAIN_BATCH_SIZE,
)

# -------------------------------------------------------------------
# model list
# -------------------------------------------------------------------
MODELS = [
    "distilbert-base-uncased",
    "roberta-base",
    "roberta-large",
    "bert-base-uncased",
    "bert-large-uncased",
    "google/electra-base-discriminator",
    "nlpaueb/sec-bert-base",
]

# per-model learning rates (anything not listed uses DEFAULT_LR)
LR_MAP = {
    "bert-base-uncased": 1e-5,
    "google/electra-base-discriminator": 1e-5,
}
DEFAULT_LR = 2e-5


def build_trainer(model, args, train_ds, val_ds, tok) -> Trainer:
    """Create the Trainer with the tokenizer attached so every saved
    checkpoint also contains the tokenizer (needed by steps 7 and 8).
    Newer transformers use `processing_class`, older ones `tokenizer`."""
    kwargs = dict(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=multi_metrics,
    )
    try:
        return Trainer(**kwargs, processing_class=tok)
    except TypeError:
        return Trainer(**kwargs, tokenizer=tok)


def finetune(ckpt: str, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
    """Fine-tune one checkpoint and save it under MODEL_DIR."""
    print(f"\n{'=' * 60}\n  {ckpt}\n{'=' * 60}")

    out_dir = Path(MODEL_DIR) / safe_name(ckpt)
    lr = LR_MAP.get(ckpt, DEFAULT_LR)

    tok = AutoTokenizer.from_pretrained(ckpt)
    train_tok = prepare_split(train_df, tok)
    val_tok = prepare_split(val_df, tok)

    model = AutoModelForSequenceClassification.from_pretrained(
        ckpt,
        num_labels=len(LABEL_COLS),
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True,
    )

    args = TrainingArguments(
        output_dir=str(out_dir),
        eval_strategy="epoch",   # older transformers: evaluation_strategy
        save_strategy="epoch",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=lr,
        logging_steps=LOGGING_STEPS,
        metric_for_best_model=METRIC_KEY,
        load_best_model_at_end=True,
        save_total_limit=SAVE_TOTAL_LIMIT,
        report_to="none",
    )

    trainer = build_trainer(model, args, train_tok, val_tok, tok)

    print(f"training (lr={lr})...")
    trainer.train()
    print("done training")

    # save best model + tokenizer at the model-folder root as well
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    print(f"best checkpoint: {trainer.state.best_model_checkpoint}")

    # free GPU memory before the next model
    del trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    data_dir = Path(DATA)
    train_df = pd.read_csv(data_dir / "train_1c.csv")
    val_df = pd.read_csv(data_dir / "val_1c.csv")

    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

    for ckpt in MODELS:
        finetune(ckpt, train_df, val_df)

    print(f"\nAll models trained. Checkpoints are in {MODEL_DIR}")
    print("Next: run 6_checkpoint_selection.py")


if __name__ == "__main__":
    main()
