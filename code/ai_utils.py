"""
ai_utils.py

Shared helpers for the transformer pipeline (steps 5-8)
=======================================================

Functions used by more than one stage live here so that fine-tuning,
checkpoint selection, performance measurement and probability
extraction all prepare data and compute metrics in exactly the same way.

Used by:

    5_transformer_finetuning.py
    6_checkpoint_selection.py
    7_performance_measuring.py
    8_probability_extraction.py

Author
------
Markus Nauroth, Southern Illinois University Edwardsville
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from transformers import AutoTokenizer

from config import LABEL_COLS

MAX_LENGTH = 512


# -------------------------------------------------------------------
# naming
# -------------------------------------------------------------------
def safe_name(ckpt: str) -> str:
    """Turn a Hugging Face model id into a folder name
    (e.g. 'nlpaueb/sec-bert-base' -> 'nlpaueb_sec-bert-base')."""
    return ckpt.replace("/", "_")


# -------------------------------------------------------------------
# data preparation
# -------------------------------------------------------------------
def df_to_hf(df):
    """Convert a pandas DataFrame into a Hugging Face Dataset
    (index is dropped so no stray index column is carried along)."""
    return Dataset.from_pandas(df.reset_index(drop=True))


def format_labels(ex):
    """Collect the label columns in LABEL_COLS into one float vector
    named 'labels', as required for multi-label classification.

    {"Risk": 1, "Governance": 0, "Strategy": 1} -> {"labels": [1.0, 0.0, 1.0]}
    """
    ex["labels"] = [float(ex[c]) for c in LABEL_COLS]
    return ex


def tokenize_fn(ex, tok, max_length: int = MAX_LENGTH):
    """Tokenize the 'text' field, truncating/padding to max_length tokens."""
    return tok(
        ex["text"],
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )


def get_torch_columns(ds):
    """Columns to hand to PyTorch. token_type_ids is only kept when the
    tokenizer produced it (BERT does; RoBERTa/DistilBERT do not)."""
    keep = ["input_ids", "attention_mask", "labels"]
    if "token_type_ids" in ds.column_names:
        keep.append("token_type_ids")
    return keep


def prepare_split(df, tok):
    """DataFrame -> labelled, tokenized, torch-formatted Dataset."""
    df.columns = df.columns.str.strip()
    ds = df_to_hf(df).map(format_labels)
    ds = ds.map(lambda e: tokenize_fn(e, tok), batched=True)
    ds.set_format("torch", columns=get_torch_columns(ds))
    return ds


# -------------------------------------------------------------------
# model / tokenizer loading
# -------------------------------------------------------------------
def load_tokenizer(ckpt_path: Path):
    """Load the tokenizer saved with a checkpoint. If the checkpoint
    folder has none (older runs), fall back to the model folder above it,
    where 5_transformer_finetuning.py also saves the tokenizer."""
    ckpt_path = Path(ckpt_path)
    try:
        return AutoTokenizer.from_pretrained(ckpt_path)
    except (OSError, ValueError):
        return AutoTokenizer.from_pretrained(ckpt_path.parent)


# -------------------------------------------------------------------
# metrics
# -------------------------------------------------------------------
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def multi_metrics(pred):
    """
    Multi-label evaluation metrics, called by the Hugging Face Trainer.

    Logits are passed through a sigmoid (labels are independent, so no
    softmax) and thresholded at 0.50.

    Returns accuracy (exact-match / subset accuracy), micro and macro
    precision, recall and F1, and macro ROC-AUC (NaN if a label has only
    one class in the evaluated split).
    """
    logits, labs = pred
    if isinstance(logits, tuple):  # some models return extra outputs
        logits = logits[0]

    probs = _sigmoid(np.asarray(logits, dtype=np.float64))
    preds = (probs >= 0.5).astype(int)
    labs = np.asarray(labs).astype(int)

    out = {"accuracy": accuracy_score(labs, preds)}
    for avg in ("micro", "macro"):
        out[f"precision_{avg}"] = precision_score(labs, preds, average=avg, zero_division=0)
        out[f"recall_{avg}"] = recall_score(labs, preds, average=avg, zero_division=0)
        out[f"f1_{avg}"] = f1_score(labs, preds, average=avg, zero_division=0)
    try:
        out["auc_macro"] = roc_auc_score(labs, probs, average="macro")
    except ValueError:
        out["auc_macro"] = float("nan")
    return out


def selection_score(f1_macro, f1_micro):
    """Checkpoint/model selection score: simple mean of macro and micro F1."""
    if f1_macro is None or f1_micro is None:
        return float("nan")
    if np.isnan(f1_macro) or np.isnan(f1_micro):
        return float("nan")
    return 0.5 * f1_macro + 0.5 * f1_micro
