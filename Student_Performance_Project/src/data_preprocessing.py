"""
data_preprocessing.py
=====================
Handles all data-loading, cleaning, encoding, and train/test splitting.
Returns ready-to-train NumPy arrays and a fitted LabelEncoder for the target.
"""

import os
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(message)s")
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
DATA_PATH   = os.path.join(DATASET_DIR, "student_data.csv")
GEN_SCRIPT  = os.path.join(DATASET_DIR, "generate_data.py")

NUMERIC_COLS = [
    "attendance_percentage",
    "study_hours_per_week",
    "assignment_score",
    "quiz_score",
    "midterm_marks",
    "previous_gpa",
]
CAT_COL = "participation_level"
TARGET  = "final_result"


# ── Public API ────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """Load CSV; auto-generate if missing."""
    if not os.path.exists(DATA_PATH):
        log.warning("Dataset not found — generating now …")
        import subprocess, sys
        subprocess.run([sys.executable, GEN_SCRIPT], check=True)

    df = pd.read_csv(DATA_PATH)
    log.info(f"Loaded {len(df)} rows × {len(df.columns)} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    1. Drop duplicate rows
    2. Fill missing numeric values with column median
    3. Fill missing categorical values with mode
    """
    before = len(df)
    df = df.drop_duplicates()
    log.info(f"Duplicates removed: {before - len(df)}")

    # Numeric → median imputation
    missing_num = df[NUMERIC_COLS].isnull().sum()
    if missing_num.any():
        log.info(f"Missing numeric values:\n{missing_num[missing_num > 0]}")
        df[NUMERIC_COLS] = df[NUMERIC_COLS].fillna(df[NUMERIC_COLS].median())

    # Categorical → mode imputation
    if df[CAT_COL].isnull().any():
        mode_val = df[CAT_COL].mode()[0]
        df[CAT_COL] = df[CAT_COL].fillna(mode_val)
        log.info(f"'{CAT_COL}' nulls filled with mode='{mode_val}'")

    # Drop rows where target is still null
    null_target = df[TARGET].isnull().sum()
    if null_target:
        log.warning(f"Dropping {null_target} rows with null target")
        df = df.dropna(subset=[TARGET])

    log.info(f"Clean shape: {df.shape}")
    return df.reset_index(drop=True)


def encode_features(df: pd.DataFrame):
    """
    Ordinal-encode participation_level (Low=0, Medium=1, High=2).
    Label-encode the target column.
    Returns (X: ndarray, y: ndarray, feature_names: list, label_encoder).
    """
    df = df.copy()

    # Ordinal map for participation
    part_map = {"Low": 0, "Medium": 1, "High": 2}
    df[CAT_COL] = df[CAT_COL].map(part_map)

    # Feature matrix
    feature_cols  = NUMERIC_COLS + [CAT_COL]
    X = df[feature_cols].values.astype(np.float32)

    # Target encoding
    le = LabelEncoder()
    y  = le.fit_transform(df[TARGET].astype(str))

    log.info(f"Classes: {list(le.classes_)}")
    log.info(f"Feature matrix shape: {X.shape}")
    return X, y, feature_cols, le


def split_data(X: np.ndarray, y: np.ndarray, test_size=0.2, random_state=42):
    """Stratified 80/20 split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    log.info(f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
    return X_train, X_test, y_train, y_test


def preprocess_pipeline():
    """Full pipeline: load → clean → encode → split. Returns all artefacts."""
    df  = load_data()
    df  = clean_data(df)
    X, y, feature_cols, le = encode_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    return X_train, X_test, y_train, y_test, feature_cols, le, df
