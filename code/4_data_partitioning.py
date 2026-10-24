"""
4_data_partitioning.py

Dataset Partitioning and Preparation Utility
============================================

This script prepares a labeled cybersecurity disclosure dataset for use in
machine learning and natural language processing experiments.

The primary purpose of the script is to:

1. Load a manually labeled training dataset.
2. Standardize column naming conventions.
3. Partition the dataset into training, validation, and test subsets.
4. Store the resulting datasets for downstream modeling tasks.

Background
----------
Machine learning models should be trained and evaluated using separate
datasets to reduce the risk of overfitting and to improve the reliability
of performance measurements.

This script implements a standard dataset partitioning strategy:

    Training Set   : 80%
    Validation Set : 10%
    Test Set       : 10%

The resulting files can be used in a variety of machine learning
workflows including:

* Traditional machine learning classification
* Transformer fine-tuning
* Large language model adaptation
* Text classification research
* Cybersecurity disclosure analysis
* SEC Item 1C disclosure studies

Input Dataset
-------------
The script expects a CSV file containing manually labeled examples.

Expected file:

    ../data/labeled_training_ai.csv

The dataset must contain a column named:

    Example Sentence

This column contains the textual data that will later be used for model
training.

Processing Steps
----------------
Step 1:
    Locate and validate the existence of the labeled dataset.

Step 2:
    Load the CSV file using pandas.

Step 3:
    Rename the text column to a standardized name ("text").

Step 4:
    Create an 80/20 split.

Step 5:
    Split the remaining 20% into validation and test sets.

Step 6:
    Store each dataset as a separate CSV file.

Output Files
------------
The following datasets are written:

train_ai.csv
    Training dataset used to fit the model.

val_ai.csv
    Validation dataset used for hyperparameter tuning and model
    selection.

test_ai.csv
    Final holdout dataset used for performance evaluation.

In addition, a combined training file is stored for quick reloading:

    PROCESSED_TRAIN_DF

Dependencies
------------
Third-party Packages:

    pandas
    scikit-learn
    tqdm

Internal Modules:

    config_ai.py

Research Context
----------------
This script is part of a cybersecurity disclosure analysis pipeline.

Typical workflow:

    1_data_acquisition.py
            ↓
    2_data_extraction_cleansing.py
            ↓
    3_data_labeling.py
            ↓
    4_data_partitioning.py
            ↓
    Model Training
            ↓
    Evaluation

Author
------
Markus Nauroth

Institution
-----------
Southern Illinois University Edwardsville

Version
-------
1.0
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pathlib import Path

# ----------------------------------------------------------------------
# Enable tqdm progress bars for pandas operations.
#
# This adds the method:
#
#     DataFrame.progress_apply(...)
#
# which behaves similarly to apply() but displays progress information.
# ----------------------------------------------------------------------

tqdm.pandas()

# ----------------------------------------------------------------------
# Optional import.
#
# Some execution environments require this import so that relative
# imports function correctly when the script is executed from the command
# line or external IDEs.
#
# The imported object is intentionally unused.
# ----------------------------------------------------------------------

try:
    import code  # noqa: F401
except ImportError:
    pass

# ----------------------------------------------------------------------
# Import project-wide configuration values.
#
# Keeping paths and filenames in a separate configuration module makes
# maintenance easier and reduces the likelihood of hardcoded paths being
# scattered throughout the codebase.
# ----------------------------------------------------------------------

from config import (
    DATA,
    DATA_DIR,
    PROCESSED_TRAIN_DF,
)

# ----------------------------------------------------------------------
# STEP 1: Locate the raw dataset.
#
# The dataset contains manually labeled examples that will later be used
# to train machine learning models.
#
# expanduser() allows the use of "~" in paths.
# resolve() converts the path into an absolute path.
# ----------------------------------------------------------------------

raw_csv_path = Path(
    "../data/labeled_training_sample_1c.csv"
).expanduser().resolve()

if not raw_csv_path.is_file():
    raise FileNotFoundError(
        f"Raw CSV not found: {raw_csv_path}"
    )

print(f"Reading raw data from: {raw_csv_path}")

# ----------------------------------------------------------------------
# STEP 2: Load the dataset.
#
# pandas is used because it provides efficient tabular data structures
# and integrates well with machine learning libraries.
# ----------------------------------------------------------------------

df = pd.read_csv(raw_csv_path, sep=",")

# ----------------------------------------------------------------------
# Verify required column existence.
#
# The downstream machine learning pipeline expects textual content.
# If this column is missing, execution stops immediately with a clear
# error message.
# ----------------------------------------------------------------------

if "Example Sentence" not in df.columns:
    raise KeyError(
        "'Example Sentence' column missing from raw file"
    )

# ----------------------------------------------------------------------
# Standardize column naming.
#
# Most NLP frameworks and custom training scripts expect the input text
# column to be named "text".
#
# Renaming here creates consistency throughout the project.
# ----------------------------------------------------------------------

df = df.rename(columns={"Example Sentence": "text"})

print(f"Loaded {len(df):,} rows")

# ----------------------------------------------------------------------
# STEP 3: Create train/validation/test partitions.
#
# First split:
#     80% Training
#     20% Temporary
#
# Second split:
#     10% Validation
#     10% Testing
#
# random_state=42 ensures experiments are reproducible.
# ----------------------------------------------------------------------

train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42
)

# ----------------------------------------------------------------------
# STEP 4: Write output datasets.
#
# Create the output directory if it does not already exist.
#
# exist_ok=True prevents an exception if the folder is already present.
# ----------------------------------------------------------------------

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)

train_path = DATA + "train_1c.csv"
val_path = DATA + "val_1c.csv"
test_path = DATA + "test_1c.csv"

# ----------------------------------------------------------------------
# Save each partition separately.
#
# index=False prevents pandas from generating an unnecessary row index
# column in the output CSV files.
# ----------------------------------------------------------------------

for sub_df, path in [
    (train_df, train_path),
    (val_df, val_path),
    (test_df, test_path),
]:

    sub_df.to_csv(path, index=False)

    print(
        f"Wrote {len(sub_df):,} rows → {path}"
    )

# ----------------------------------------------------------------------
# Store a separate combined training dataset.
#
# Some downstream workflows expect a single training file rather than
# separate train/validation/test sets.
#
# This file provides a convenient reload point for future experiments.
# ----------------------------------------------------------------------

train_df.to_csv(
    PROCESSED_TRAIN_DF,
    index=False
)

print(
    f"\nAll done. Combined training file saved to "
    f"{PROCESSED_TRAIN_DF}"
)