"""
predict.py
==========
Interactive CLI: prompts the user for student feature values and returns
a predicted grade category using the saved best model.

Usage:
    python src/predict.py
"""

import os
import sys
import logging
import numpy as np
import joblib

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(message)s")

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")

# ── Feature metadata ──────────────────────────────────────────────────────────
FEATURES = [
    {
        "key":   "attendance_percentage",
        "label": "Attendance percentage     (0–100)",
        "type":  float,
        "min":   0.0, "max": 100.0,
    },
    {
        "key":   "study_hours_per_week",
        "label": "Study hours per week      (1–40)",
        "type":  float,
        "min":   1.0, "max": 40.0,
    },
    {
        "key":   "assignment_score",
        "label": "Assignment score          (0–100)",
        "type":  float,
        "min":   0.0, "max": 100.0,
    },
    {
        "key":   "quiz_score",
        "label": "Quiz score               (0–100)",
        "type":  float,
        "min":   0.0, "max": 100.0,
    },
    {
        "key":   "midterm_marks",
        "label": "Midterm marks            (0–100)",
        "type":  float,
        "min":   0.0, "max": 100.0,
    },
    {
        "key":   "previous_gpa",
        "label": "Previous GPA             (0.0–4.0)",
        "type":  float,
        "min":   0.0, "max": 4.0,
    },
    {
        "key":   "participation_level",
        "label": "Participation level      [Low / Medium / High]",
        "type":  str,
        "choices": {"low": 0, "medium": 1, "high": 2},
    },
]

PART_MAP = {"low": 0, "medium": 1, "high": 2}


def _prompt_numeric(feat: dict) -> float:
    while True:
        try:
            val = float(input(f"  {feat['label']}: ").strip())
            if feat["min"] <= val <= feat["max"]:
                return val
            print(f"    ✗  Value must be between {feat['min']} and {feat['max']}.")
        except ValueError:
            print("    ✗  Please enter a numeric value.")


def _prompt_participation() -> int:
    while True:
        val = input("  Participation level      [Low / Medium / High]: ").strip().lower()
        if val in PART_MAP:
            return PART_MAP[val]
        print("    ✗  Enter one of: Low, Medium, High")


def collect_input() -> np.ndarray:
    """Interactively collect one student record. Returns shape (1, 7)."""
    print("\n" + "═" * 50)
    print("  Student Performance Predictor")
    print("  Enter the student's details below:")
    print("═" * 50)

    row = []
    for feat in FEATURES:
        if feat["type"] == str:
            row.append(_prompt_participation())
        else:
            row.append(_prompt_numeric(feat))

    return np.array(row, dtype=np.float32).reshape(1, -1)


def predict_single(X: np.ndarray, model, label_encoder=None) -> str:
    """Return the predicted class label (string)."""
    y_pred = model.predict(X)[0]
    if label_encoder is not None:
        return label_encoder.inverse_transform([y_pred])[0]
    # If no encoder stored, return raw integer
    return str(y_pred)


def main():
    # ── Load model ────────────────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        print(f"\n  ✗  No trained model found at:\n     {MODEL_PATH}")
        print("  Run 'python app.py' first to train and save the model.\n")
        sys.exit(1)

    model = joblib.load(MODEL_PATH)

    # Try to load label encoder stored alongside model
    le_path = MODEL_PATH.replace(".pkl", "_le.pkl")
    le = joblib.load(le_path) if os.path.exists(le_path) else None

    while True:
        X = collect_input()
        label = predict_single(X, model, le)

        print("\n" + "─" * 50)
        print(f"  ★  Predicted Grade Category:  {label}")
        print("─" * 50 + "\n")

        again = input("  Predict another student? [y/N]: ").strip().lower()
        if again != "y":
            print("\n  Goodbye!\n")
            break


if __name__ == "__main__":
    main()
