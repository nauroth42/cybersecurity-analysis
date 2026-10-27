"""
6_checkpoint_selection.py

Checkpoint Selection
====================

Purpose
-------
Reads the trainer_state.json of every saved checkpoint produced by
5_transformer_finetuning.py, scores each checkpoint on its VALIDATION
metrics, keeps the best checkpoint per model, and ranks the models.

Score
-----
    score = 0.5 * eval_f1_macro + 0.5 * eval_f1_micro

Only validation metrics are used, so the test set stays untouched until
7_performance_measuring.py.

Input
-----
    MODEL_DIR/<model>/checkpoint-<step>/trainer_state.json

Output
------
    CHECKPOINT_METRICS_CSV
        One row per model (its best checkpoint), sorted best first,
        with a 'rank' column. Row 1 is the overall best model, which
        8_probability_extraction.py uses.

    <CHECKPOINT_METRICS_CSV stem>_all.csv
        Every scored checkpoint, for transparency/reporting.

Configuration (config_ai.py)
----------------------------
    MODEL_DIR, CHECKPOINT_METRICS_CSV

Author
------
Markus Nauroth, Southern Illinois University Edwardsville

Version
-------
1.1
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ai_utils import selection_score
from config import CHECKPOINT_METRICS_CSV, MODEL_DIR


def checkpoint_step(ckpt_dir: Path) -> int | None:
    """'checkpoint-1250' -> 1250"""
    try:
        return int(ckpt_dir.name.split("-")[-1])
    except ValueError:
        return None


def eval_entry_for_checkpoint(state_file: Path, step: int) -> dict | None:
    """
    Return the evaluation log entry belonging to this checkpoint.

    trainer_state.json contains the full log history up to that point,
    so later checkpoints also hold the evaluations of earlier epochs.
    Only the entry whose step equals the checkpoint's step describes
    this checkpoint's weights.
    """
    with state_file.open(encoding="utf-8") as f:
        js = json.load(f)

    evals = [
        e for e in js.get("log_history", [])
        if any(k.startswith("eval_") for k in e)
    ]
    for e in evals:
        if e.get("step") == step:
            return e
    return None


def collect_checkpoints(model_dir: Path) -> pd.DataFrame:
    records: list[dict] = []
    for model_folder in sorted(model_dir.iterdir()):
        if not model_folder.is_dir():
            continue
        for ckpt in sorted(model_folder.glob("checkpoint-*")):
            state_file = ckpt / "trainer_state.json"
            step = checkpoint_step(ckpt)
            if step is None or not state_file.is_file():
                continue
            entry = eval_entry_for_checkpoint(state_file, step)
            if entry is None:
                print(f"  no eval entry for {model_folder.name}/{ckpt.name}, skipped")
                continue
            records.append({
                "model": model_folder.name,
                "checkpoint": ckpt.name,
                **entry,
                "state_path": str(state_file),
            })
    return pd.DataFrame.from_records(records)


def main() -> None:
    model_dir = Path(MODEL_DIR)
    df = collect_checkpoints(model_dir)
    if df.empty:
        raise SystemExit(f"No checkpoints with evaluation logs found in {model_dir}/*/checkpoint-*")

    df["score"] = [
        selection_score(r.get("eval_f1_macro"), r.get("eval_f1_micro"))
        for r in df.to_dict("records")
    ]

    out_csv = Path(CHECKPOINT_METRICS_CSV)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    all_csv = out_csv.with_name(f"{out_csv.stem}_all{out_csv.suffix}")
    df.sort_values(["model", "step"]).to_csv(all_csv, index=False)

    best = (
        df.dropna(subset=["score"])
          .sort_values("score", ascending=False)
          .groupby("model", as_index=False)
          .first()
          .sort_values("score", ascending=False)
          .reset_index(drop=True)
    )
    best.insert(0, "rank", range(1, len(best) + 1))
    best.to_csv(out_csv, index=False)

    print(best[["rank", "model", "checkpoint", "score"]].to_string(index=False))
    print(f"\nWrote {len(best)} best checkpoints -> {out_csv}")
    print(f"Wrote all {len(df)} scored checkpoints -> {all_csv}")
    print("Next: run 7_performance_measuring.py")


if __name__ == "__main__":
    main()
