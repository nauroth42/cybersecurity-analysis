# 02_train_models.py
# Trains eight transformer checkpoints on the multi-label task and
# writes a comparison table of validation / test metrics.
# In the console, execute
# - pip install transformers
# - pip install torch
# - pip install datasets
# - pip install accelerate
# - pip install scikit-learn

import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from config_ai import (
    DATA,
    MODEL,
    MODEL_DIR,
    MODEL_COMPARISON_CSV,
    LABEL_COLS,
    NUM_EPOCHS,
    TRAIN_BATCH_SIZE,
    EVAL_BATCH_SIZE,
    LOGGING_STEPS,
    METRIC_KEY,
    SAVE_TOTAL_LIMIT,
)

tqdm.pandas()  # enable df.progress_apply if needed

# -------------------------------------------------------------------
# helpers
# -------------------------------------------------------------------
def df_to_hf(df):
    return Dataset.from_pandas(df.reset_index(drop=True))


def format_labels(ex):
    ex["labels"] = [float(ex[c]) for c in LABEL_COLS]
    return ex


def tokenize_fn(ex, tok):
    return tok(
        ex["text"],
        max_length=512,
        truncation=True,
        padding="max_length",
    )


def get_torch_columns(ds):
    """
    FIX 2: only keep columns the model actually accepts.
    RoBERTa and DistilBERT do not produce token_type_ids;
    including it causes a shape error inside the Trainer.
    """
    keep = ["input_ids", "attention_mask", "labels"]
    if "token_type_ids" in ds.column_names:
        keep.append("token_type_ids")
    return keep


def multi_metrics(pred):
    logits, labs = pred
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs >= 0.5).astype(int)
    labs = labs.astype(int)

    acc = accuracy_score(labs, preds)
    p_micro = precision_score(labs, preds, average="micro", zero_division=0)
    r_micro = recall_score(labs, preds, average="micro", zero_division=0)
    f_micro = f1_score(labs, preds, average="micro", zero_division=0)
    p_macro = precision_score(labs, preds, average="macro", zero_division=0)
    r_macro = recall_score(labs, preds, average="macro", zero_division=0)
    f_macro = f1_score(labs, preds, average="macro", zero_division=0)
    try:
        auc_macro = roc_auc_score(labs, probs, average="macro")
    except ValueError:
        auc_macro = float("nan")

    return {
        "accuracy": acc,
        "precision_micro": p_micro,
        "recall_micro": r_micro,
        "f1_micro": f_micro,
        "precision_macro": p_macro,
        "recall_macro": r_macro,
        "f1_macro": f_macro,
        "auc_macro": auc_macro,
    }


# -------------------------------------------------------------------
# data
# -------------------------------------------------------------------
train_df = pd.read_csv(DATA + "train_ai.csv")
val_df = pd.read_csv(DATA + "val_ai.csv")
test_df = pd.read_csv(DATA + "test_ai.csv")

train_ds = df_to_hf(train_df).map(format_labels)
val_ds = df_to_hf(val_df).map(format_labels)
test_ds = df_to_hf(test_df).map(format_labels)

# -------------------------------------------------------------------
# model list
# -------------------------------------------------------------------
MODELS = [
    "microsoft/deberta-v3-base",
    "microsoft/deberta-v3-large",
    "microsoft/deberta-v3-xl",
    "microsoft/deberta-v3-xl-large",
    "yiyanghkust/finbert-pretrain",
    "MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
    "roberta-base",
    "roberta-large",
    "facebook/roberta-xl",
    "bert-base-uncased",
    "distilbert-base-uncased",
]

# per-model learning rates: DeBERTa-v3 needs a lower lr than BERT/RoBERTa
LR_MAP = {
    "microsoft/deberta-v3-base":                   2e-5,
    "microsoft/deberta-v3-large":                  1e-5,
    "MoritzLaurer/deberta-v3-base-zeroshot-v2.0":  2e-5,
    "yiyanghkust/finbert-pretrain":                2e-5,
    "facebook/roberta-xl":                         1e-5,
}
DEFAULT_LR = 2e-5
# -------------------------------------------------------------------
# training loop
# -------------------------------------------------------------------
"""
MODEL_DIR.mkdir(parents=True, exist_ok=True)
summary_rows = []

for ckpt in MODELS:
    print(f"\n{'='*60}\n  {ckpt}\n{'='*60}")
    out_dir = MODEL + ckpt.replace("/", "_")
    tok = AutoTokenizer.from_pretrained(ckpt)

    train_tok = train_ds.map(lambda e: tokenize_fn(e, tok), batched=True)
    val_tok = val_ds.map(lambda e: tokenize_fn(e, tok), batched=True)
    test_tok = test_ds.map(lambda e: tokenize_fn(e, tok), batched=True)

    for ds in (train_tok, val_tok, test_tok):
        ds.set_format(
            "torch",
            columns=["input_ids", "attention_mask", "labels"],
        )

    model = AutoModelForSequenceClassification.from_pretrained(
        ckpt,
        num_labels=len(LABEL_COLS),
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True,
    )

    args = TrainingArguments(
        output_dir=out_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        logging_steps=LOGGING_STEPS,
        metric_for_best_model=METRIC_KEY,
        load_best_model_at_end=True,
        save_total_limit=SAVE_TOTAL_LIMIT,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tok,
        compute_metrics=multi_metrics,
    )

    print('training...')
    trainer.train()
    print('trained')

    val_metrics = trainer.evaluate(eval_dataset=val_tok)
    test_metrics = trainer.evaluate(eval_dataset=test_tok)

    row = {
        "model": ckpt,
        **{f"val_{k}": v for k, v in val_metrics.items()},
        **{f"test_{k}": v for k, v in test_metrics.items()},
    }
    print(row)
    summary_rows.append(row)
"""

# FIX 4: use MODEL_DIR consistently for mkdir; derive out_dir from it
MODEL_DIR.mkdir(parents=True, exist_ok=True)
summary_rows = []

for ckpt in MODELS:
    print(f"\n{'=' * 60}\n  {ckpt}\n{'=' * 60}")

    out_dir = str(MODEL_DIR / ckpt.replace("/", "_"))
    lr = LR_MAP.get(ckpt, DEFAULT_LR)

    tok = AutoTokenizer.from_pretrained(ckpt)

    train_tok = train_ds.map(lambda e: tokenize_fn(e, tok), batched=True)
    val_tok = val_ds.map(lambda e: tokenize_fn(e, tok), batched=True)
    test_tok = test_ds.map(lambda e: tokenize_fn(e, tok), batched=True)

    # FIX 2: set only the columns the model accepts (no stray token_type_ids)
    for ds in (train_tok, val_tok, test_tok):
        ds.set_format("torch", columns=get_torch_columns(train_tok))

    model = AutoModelForSequenceClassification.from_pretrained(
        ckpt,
        num_labels=len(LABEL_COLS),
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True,
    )

    args = TrainingArguments(
        output_dir=out_dir,
        evaluation_strategy="epoch",  # FIX 5: "eval_strategy" was renamed;
        save_strategy="epoch",  # "evaluation_strategy" is the stable key
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=lr,  # FIX 6: explicit per-model lr
        logging_steps=LOGGING_STEPS,
        metric_for_best_model=METRIC_KEY,
        load_best_model_at_end=True,
        save_total_limit=SAVE_TOTAL_LIMIT,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tok,  # FIX 7: uncommented — required so the
        compute_metrics=multi_metrics,  # best checkpoint is saved with the tokenizer
    )

    print("training...")
    trainer.train()
    print("done training")

    val_metrics = trainer.evaluate(eval_dataset=val_tok)
    test_metrics = trainer.evaluate(eval_dataset=test_tok)

    row = {
        "model": ckpt,
        **{f"val_{k}": v for k, v in val_metrics.items()},
        **{f"test_{k}": v for k, v in test_metrics.items()},
    }
    print(row)
    summary_rows.append(row)

# -------------------------------------------------------------------
# write comparison table
# -------------------------------------------------------------------
pd.DataFrame(summary_rows).to_csv(MODEL_COMPARISON_CSV, index=False)
print(f"\nSaved comparison metrics → {MODEL_COMPARISON_CSV}")