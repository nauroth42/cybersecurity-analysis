# 04_infer_documents.py
# Loads the best checkpoint chosen in 03_pick_checkpoints.py, scores every
# .txt file in data/item_1c, and writes a tidy CSV with probabilities and
# binary predictions.

import csv
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import sigmoid
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config_ai import (
    INFERENCE_TEXT_DIR,
    INFERENCE_PARTIAL_CSV,
    INFERENCE_OUT_CSV,
    CHECKPOINT_METRICS_CSV,
    MODEL_DIR,
    LABEL_COLS,
    CHUNK_SIZE,
    AGGREGATION_METHOD,
    THRESHOLD,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----------------------------------------------------------
# 1. Load the best checkpoint path from the metrics CSV
# ----------------------------------------------------------
df_best = pd.read_csv(CHECKPOINT_METRICS_CSV)
if df_best.empty:
    raise SystemExit("No best-checkpoint table found; run 03_pick_checkpoints.py first.")

row = df_best.iloc[0]
model_name = row["model"]
ckpt_folder = row["checkpoint"]
ckpt_path = MODEL_DIR / model_name / ckpt_folder
print(f"Scoring with {model_name} → {ckpt_folder}")

# ----------------------------------------------------------
# 2. Load tokenizer and model
# ----------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
model = AutoModelForSequenceClassification.from_pretrained(
    ckpt_path, ignore_mismatched_sizes=True
).to(device)
model.eval()

# ----------------------------------------------------------
# 3. Determine which files still need scoring (skip logic)
# ----------------------------------------------------------
INFERENCE_TEXT_DIR.mkdir(parents=True, exist_ok=True)
all_txt = sorted([p for p in INFERENCE_TEXT_DIR.glob("*.txt")])

done = set()
if Path(INFERENCE_PARTIAL_CSV).is_file():
    done = set(pd.read_csv(INFERENCE_PARTIAL_CSV)["filename"].unique())

to_score = [p for p in all_txt if p.name not in done]
print(f"{len(all_txt)} total files, {len(done)} already done, {len(to_score)} new.")

# ----------------------------------------------------------
# 4. Helper to score one document
# ----------------------------------------------------------
def score_file(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore")
    tokens = tokenizer.encode(text, add_special_tokens=False)

    chunk_probs = []
    for start in range(0, len(tokens), CHUNK_SIZE):
        chunk = tokens[start : start + CHUNK_SIZE]
        input_ids = tokenizer.build_inputs_with_special_tokens(chunk)
        ids = torch.tensor([input_ids], dtype=torch.long).to(device)
        mask = torch.ones_like(ids).to(device)

        with torch.no_grad():
            probs = sigmoid(model(ids, attention_mask=mask).logits)[0].cpu().numpy()
        chunk_probs.append(probs)

    if not chunk_probs:
        agg_avg = np.zeros(len(LABEL_COLS))
        agg_max = np.zeros(len(LABEL_COLS))
    else:
        stack = np.vstack(chunk_probs)
        agg_avg = stack.mean(0)
        agg_max = stack.max(0)

    preds_avg = (agg_avg >= THRESHOLD).astype(int)
    preds_max = (agg_max >= THRESHOLD).astype(int)
    row = {"filename": path.name}
    for i, lbl in enumerate(LABEL_COLS):
        row[f"{lbl}_prob_avg"] = round(float(agg_avg[i]), 6)
        row[f"{lbl}_pred_avg"] = int(preds_avg[i])
        row[f"{lbl}_prob_max"] = round(float(agg_max[i]), 6)
        row[f"{lbl}_pred_max"] = int(preds_max[i])
    return row


# ----------------------------------------------------------
# 5. Stream results to the partial CSV
# ----------------------------------------------------------
mode = "a" if Path(INFERENCE_PARTIAL_CSV).is_file() else "w"
with open(INFERENCE_PARTIAL_CSV, mode, newline="", encoding="utf-8") as f_out:
    writer = None
    for p in tqdm(to_score, desc="Scoring files"):
        res = score_file(p)
        if writer is None:
            writer = csv.DictWriter(f_out, fieldnames=list(res.keys()))
            if mode == "w":
                writer.writeheader()
        writer.writerow(res)
        f_out.flush()

print("Partial results saved.")

# ----------------------------------------------------------
# 6. De-duplicate and write final CSV
# ----------------------------------------------------------
df_final = pd.read_csv(INFERENCE_PARTIAL_CSV).drop_duplicates("filename", keep="last")
df_final.to_csv(INFERENCE_OUT_CSV, index=False)
print(f"Final inference table → {INFERENCE_OUT_CSV}")
