"""
8_prob_extraction.py

Probability Extraction and Large-Scale Inference Framework
==========================================================

Purpose
-------
This script applies a previously fine-tuned transformer model to a
large collection of unseen documents and extracts prediction
probabilities for every target category.

Unlike model training scripts, this program focuses on production-scale
inference and probability generation.

The resulting probability estimates can be used for:

* Cybersecurity disclosure analysis
* SEC Item 1C classification
* Risk disclosure measurement
* Governance analysis
* Strategy disclosure analysis
* Regulatory compliance research
* Longitudinal disclosure studies
* Financial text analytics

Research Context
----------------
This script represents one of the final stages in the machine learning
pipeline.

Overall Workflow
----------------

    1_data_acquisition.py
            ↓
    2_data_extraction_cleansing.py
            ↓
    3_data_labeling.py
            ↓
    4_data_partitioning.py
            ↓
    5_transformer_finetuning.py
            ↓
    6_checkpoint_selection.py
            ↓
    7_performance_measuring.py
            ↓
    8_prob_extraction.py

The script loads the best-performing checkpoint selected during model
evaluation and applies it to previously unseen text documents.

Objectives
----------
The system is designed to:

1. Load the best checkpoint identified during validation.
2. Read all available inference documents.
3. Skip documents already processed.
4. Generate probabilities for each label.
5. Convert probabilities into binary predictions.
6. Store partial results for fault tolerance.
7. Produce a consolidated prediction dataset.

Input Files
-----------
Checkpoint Selection File
~~~~~~~~~~~~~~~~~~~~~~~~~

Produced by:

    6_checkpoint_selection.py

Contains:

    * Model name
    * Checkpoint identifier
    * Performance statistics

Inference Documents
~~~~~~~~~~~~~~~~~~~

Directory:

    data/item_1c/

Typical contents:

    company_001.txt
    company_002.txt
    company_003.txt

Output Files
------------
Partial Output
~~~~~~~~~~~~~~

Used for fault-tolerant processing:

    inference_partial.csv

Final Output
~~~~~~~~~~~~

Contains final document classifications:

    inference_results.csv

Output Structure
----------------

Each row typically contains:

    filename
    probability_<label>
    prediction_<label>

Example:

    filename
    probability_governance
    prediction_governance
    probability_strategy
    prediction_strategy

Chunking Strategy
-----------------
Transformer architectures generally support a maximum sequence length
of approximately 512 tokens.

Since SEC filings and cybersecurity disclosures frequently exceed this
limit, the script divides longer documents into smaller segments.

Each segment is independently evaluated and subsequently aggregated.

Aggregation Methods
-------------------

The aggregation method is controlled through:

    AGGREGATION_METHOD

Possible approaches include:

Mean Aggregation
    Average probability across all chunks.

Maximum Aggregation
    Use highest observed probability.

Weighted Aggregation
    Weight probabilities according to chunk size.

Hardware Support
----------------
The script automatically detects available hardware.

Supported execution environments:

* NVIDIA CUDA GPU
* CPU

When a CUDA-compatible GPU is available, inference speed is
substantially improved.

Dependencies
------------
Python Standard Library

* csv
* math
* pathlib

Third-Party Libraries

* pandas
* numpy
* torch
* transformers
* tqdm

Author
------
Markus Nauroth

Institution
-----------
Southern Illinois University Edwardsville

Research Areas
--------------
* Cybersecurity
* Artificial Intelligence
* Natural Language Processing
* Financial Disclosure Analysis

Version
-------
1.0
"""

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
def main() -> None:
    """
    Execute the complete probability extraction workflow.

    This function orchestrates the end-to-end inference process,
    including model loading, checkpoint recovery, document
    selection, probability generation, intermediate persistence,
    and final result creation.

    Returns
    -------
    None

    Workflow
    --------
    1. Load best checkpoint information.
    2. Initialize tokenizer.
    3. Initialize transformer model.
    4. Identify unprocessed files.
    5. Execute inference.
    6. Save intermediate results.
    7. Merge and deduplicate outputs.
    8. Save final prediction table.

    Fault Tolerance
    ---------------
    Intermediate prediction results are continuously written to
    disk to minimize data loss if execution is interrupted.

    This is especially valuable when processing thousands of SEC
    filings or large collections of cybersecurity disclosures.
    """

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
