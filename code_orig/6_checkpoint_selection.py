# 03_pick_checkpoints.py
# Reads trainer_state.json files, scores every checkpoint, keeps the best
# per model, and writes a tidy CSV that later scripts can load for inference.

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from config_item1c import MODEL_DIR, CHECKPOINT_METRICS_CSV

# ----------------------------------------------------------------------
# Helper – parse one trainer_state.json
# ----------------------------------------------------------------------
def parse_trainer_state(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        js = json.load(f)
    rows = []
    for entry in js.get("log_history", []):
        if any(k.startswith("eval_") for k in entry):
            rows.append(entry)
    return rows


# ----------------------------------------------------------------------
# Walk through every model directory
# ----------------------------------------------------------------------
records: list[dict] = []

for model_folder in MODEL_DIR.iterdir():  # e.g. models/bert-base-uncased
    if not model_folder.is_dir():
        continue
    ckpts = [d for d in model_folder.iterdir() if d.name.startswith("checkpoint-")]
    for ckpt in ckpts:
        state_file = ckpt / "trainer_state.json"
        if not state_file.is_file():
            continue
        for row in parse_trainer_state(state_file):
            row.update({
                "model": model_folder.name,
                "checkpoint": ckpt.name,
                "state_path": str(state_file),
            })
            records.append(row)

if not records:
    raise SystemExit("No trainer_state.json files found in models/*/checkpoint-*")

df = pd.DataFrame.from_records(records)

# ----------------------------------------------------------------------
# Score rows: simple average of f1_macro and f1_micro
# ----------------------------------------------------------------------
def score(r):
    if pd.isna(r["eval_f1_macro"]) or pd.isna(r["eval_f1_micro"]):
        return np.nan
    return 0.5 * r["eval_f1_macro"] + 0.5 * r["eval_f1_micro"]

df["score"] = df.apply(score, axis=1)

# ----------------------------------------------------------------------
# Pick the top-scoring checkpoint for each model
# ----------------------------------------------------------------------
best_rows = (
    df.sort_values("score", ascending=False)
      .dropna(subset=["score"])
      .groupby("model", as_index=False)
      .first()
)

best_rows.to_csv(CHECKPOINT_METRICS_CSV, index=False)
print(f"Wrote {len(best_rows)} best rows → {CHECKPOINT_METRICS_CSV}")
