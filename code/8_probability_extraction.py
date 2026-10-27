"""
8_probability_extraction.py

Probability Extraction (Inference on Unseen Documents)
======================================================

Purpose
-------
Applies the overall best checkpoint (rank 1 in CHECKPOINT_METRICS_CSV,
from 6_checkpoint_selection.py) to a folder of unseen text documents and
stores per-label probabilities and binary predictions.

Long documents
--------------
Documents longer than the model's input limit are split into token
chunks. Each chunk is scored, then chunk probabilities are aggregated
two ways, both written to the output:

    *_prob_avg / *_pred_avg   mean over chunks
    *_prob_max / *_pred_max   maximum over chunks

Predictions use: probability >= THRESHOLD.

Resumable runs
--------------
Results are streamed to INFERENCE_PARTIAL_CSV after every file. Files
already listed there are skipped on the next run, so an interrupted run
can simply be restarted.

Input
-----
    CHECKPOINT_METRICS_CSV        from step 6
    INFERENCE_TEXT_DIR/*.txt

Output
------
    INFERENCE_PARTIAL_CSV         running results
    INFERENCE_OUT_CSV             final, de-duplicated results

Configuration (config_ai.py)
----------------------------
    INFERENCE_TEXT_DIR, INFERENCE_PARTIAL_CSV, INFERENCE_OUT_CSV,
    CHECKPOINT_METRICS_CSV, MODEL_DIR, LABEL_COLS, CHUNK_SIZE, THRESHOLD

Hardware
--------
Uses a CUDA GPU when available, otherwise the CPU.

Author
------
Markus Nauroth, Southern Illinois University Edwardsville

Version
-------
1.1
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification

from ai_utils import MAX_LENGTH, load_tokenizer
from config import (
    CHECKPOINT_METRICS_CSV,
    CHUNK_SIZE,
    INFERENCE_OUT_CSV,
    INFERENCE_PARTIAL_CSV,
    INFERENCE_TEXT_DIR,
    LABEL_COLS,
    MODEL_DIR,
    THRESHOLD,
)


def best_checkpoint_path() -> tuple[str, Path]:
    """Path of the overall best checkpoint (highest selection score)."""
    df_best = pd.read_csv(CHECKPOINT_METRICS_CSV)
    if df_best.empty:
        raise SystemExit("No best-checkpoint table found; run 6_checkpoint_selection.py first.")
    row = df_best.sort_values("score", ascending=False).iloc[0]
    return row["model"], Path(MODEL_DIR) / row["model"] / row["checkpoint"]


def num_special_tokens(tokenizer) -> int:
    """Number of special tokens added to one sequence ([CLS]/[SEP], <s>/</s>, ...)."""
    try:
        return tokenizer.num_special_tokens_to_add(pair=False)
    except AttributeError:  # not available in some transformers versions
        return len(tokenizer("")["input_ids"])


def effective_chunk_size(tokenizer) -> int:
    """CHUNK_SIZE, capped so chunk + special tokens fits the model."""
    max_len = min(tokenizer.model_max_length or MAX_LENGTH, MAX_LENGTH)
    limit = max_len - num_special_tokens(tokenizer)
    if CHUNK_SIZE > limit:
        print(f"CHUNK_SIZE {CHUNK_SIZE} too large for this model; using {limit}.")
    return min(CHUNK_SIZE, limit)


def chunk_input_ids(text: str, tokenizer, chunk_size: int) -> list[list[int]]:
    """
    Split a document into consecutive chunks of at most chunk_size tokens,
    each wrapped in the model's special tokens.

    Uses the tokenizer's own overflow handling (works in transformers 4.x
    and 5.x, where build_inputs_with_special_tokens no longer exists).
    Falls back to manual chunking for slow tokenizers that lack overflow support.
    """
    if not text.strip():
        return []
    try:
        enc = tokenizer(
            text,
            max_length=chunk_size + num_special_tokens(tokenizer),
            truncation=True,
            stride=0,
            return_overflowing_tokens=True,
            padding=False,
        )
        ids = enc["input_ids"]
        # a single chunk can come back as a flat list
        return ids if ids and isinstance(ids[0], list) else [ids]
    except (NotImplementedError, ValueError, TypeError):
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return [
            tokenizer.build_inputs_with_special_tokens(tokens[i : i + chunk_size])
            for i in range(0, len(tokens), chunk_size)
        ]


def score_file(path: Path, tokenizer, model, device, chunk_size: int) -> dict:
    """
    Multi-label inference on one document.

    1. Read and tokenize the text.
    2. Split tokens into chunks of chunk_size.
    3. Score each chunk (sigmoid on logits; labels are independent).
    4. Aggregate chunk probabilities by mean and by max.
    5. Threshold to binary predictions.

    Returns a dict with 'filename' and, per label,
    <label>_prob_avg, <label>_pred_avg, <label>_prob_max, <label>_pred_max.
    """
    text = path.read_text(encoding="utf-8", errors="ignore")

    chunk_probs = []
    for input_ids in chunk_input_ids(text, tokenizer, chunk_size):
        ids = torch.tensor([input_ids], dtype=torch.long, device=device)
        mask = torch.ones_like(ids)
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=mask).logits
        chunk_probs.append(torch.sigmoid(logits)[0].cpu().numpy())

    if chunk_probs:
        stack = np.vstack(chunk_probs)
        agg_avg, agg_max = stack.mean(0), stack.max(0)
    else:  # empty document
        agg_avg = agg_max = np.zeros(len(LABEL_COLS))

    preds_avg = (agg_avg >= THRESHOLD).astype(int)
    preds_max = (agg_max >= THRESHOLD).astype(int)

    row = {"filename": path.name}
    for i, lbl in enumerate(LABEL_COLS):
        row[f"{lbl}_prob_avg"] = round(float(agg_avg[i]), 6)
        row[f"{lbl}_pred_avg"] = int(preds_avg[i])
        row[f"{lbl}_prob_max"] = round(float(agg_max[i]), 6)
        row[f"{lbl}_pred_max"] = int(preds_max[i])
    return row


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1. best checkpoint
    model_name, ckpt_path = best_checkpoint_path()
    print(f"Scoring with {model_name} -> {ckpt_path.name}")

    # 2. tokenizer and model
    tokenizer = load_tokenizer(ckpt_path)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_path).to(device)
    model.eval()
    chunk_size = effective_chunk_size(tokenizer)

    # 3. which files still need scoring
    text_dir = Path(INFERENCE_TEXT_DIR)
    text_dir.mkdir(parents=True, exist_ok=True)
    all_txt = sorted(text_dir.glob("*.txt"))

    partial = Path(INFERENCE_PARTIAL_CSV)
    done = set()
    if partial.is_file() and partial.stat().st_size > 0:
        done = set(pd.read_csv(partial)["filename"].unique())
    to_score = [p for p in all_txt if p.name not in done]
    print(f"{len(all_txt)} total files, {len(done)} already done, {len(to_score)} new.")

    # 4. stream results to the partial CSV
    partial.parent.mkdir(parents=True, exist_ok=True)
    write_header = not (partial.is_file() and partial.stat().st_size > 0)
    with partial.open("a", newline="", encoding="utf-8") as f_out:
        writer = None
        for p in tqdm(to_score, desc="Scoring files"):
            res = score_file(p, tokenizer, model, device, chunk_size)
            if writer is None:
                writer = csv.DictWriter(f_out, fieldnames=list(res.keys()))
                if write_header:
                    writer.writeheader()
            writer.writerow(res)
            f_out.flush()
    print("Partial results saved.")

    # 5. de-duplicate and write final CSV
    if not partial.is_file() or partial.stat().st_size == 0:
        raise SystemExit(f"No documents scored; is {text_dir} empty?")
    df_final = pd.read_csv(partial).drop_duplicates("filename", keep="last")
    df_final.to_csv(INFERENCE_OUT_CSV, index=False)
    print(f"Final inference table -> {INFERENCE_OUT_CSV}")


if __name__ == "__main__":
    main()
