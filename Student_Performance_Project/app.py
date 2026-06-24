"""
app.py
======
Main entry point for the Student Performance Prediction System.

Workflow:
  1. Load & clean dataset
  2. Exploratory Data Analysis → save charts
  3. Encode features / split data
  4. Train Decision Tree, Random Forest, Naïve Bayes
  5. Evaluate & compare all models
  6. Save best model + label encoder to models/
  7. Optional: interactive single-student prediction

Run:
    python app.py
"""

import os
import sys
import logging
import joblib

# ── Project root on sys.path ──────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

from src.data_preprocessing import preprocess_pipeline
from src.visualization      import run_eda, plot_model_comparison
from src.train_model        import train_and_select
from src.evaluate           import evaluate_all, print_summary_table
from src.predict            import collect_input, predict_single

MODELS_DIR = os.path.join(ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
LE_PATH    = MODEL_PATH.replace(".pkl", "_le.pkl")
os.makedirs(MODELS_DIR, exist_ok=True)


def banner(text: str, width=58):
    print("\n" + "═" * width)
    print(f"  {text}")
    print("═" * width)


def main():
    banner("Student Performance Prediction System  v1.0")

    # ── 1. Pre-process ────────────────────────────────────────────────────────
    banner("Step 1 │ Load & Pre-process Data")
    X_train, X_test, y_train, y_test, feature_cols, le, df = preprocess_pipeline()
    log.info(f"Features used: {feature_cols}")

    # ── 2. EDA ────────────────────────────────────────────────────────────────
    banner("Step 2 │ Exploratory Data Analysis")
    run_eda(df)

    # ── 3. Train ──────────────────────────────────────────────────────────────
    banner("Step 3 │ Train Models")
    train_result = train_and_select(X_train, X_test, y_train, y_test)
    models    = train_result["models"]
    best_name = train_result["best_name"]
    results   = train_result["results"]

    # ── 4. Evaluate ───────────────────────────────────────────────────────────
    banner("Step 4 │ Evaluate All Models")
    summary = evaluate_all(models, X_test, y_test, le)
    print_summary_table(summary)

    # ── 5. Model comparison chart ─────────────────────────────────────────────
    plot_model_comparison(results)

    # ── 6. Persist best model + label encoder ─────────────────────────────────
    banner("Step 5 │ Save Best Model")
    best_model = models[best_name]
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(le,         LE_PATH)
    print(f"\n  ★  Best model  : {best_name}")
    print(f"     Accuracy    : {results[best_name]*100:.2f} %")
    print(f"     Saved to    : {MODEL_PATH}")

    # ── 7. Interactive prediction (optional) ──────────────────────────────────
    banner("Step 6 │ Interactive Prediction (optional)")
    try:
        run_predict = input("\n  Would you like to predict a student's grade now? [y/N]: ")
    except EOFError:
        run_predict = "n"

    if run_predict.strip().lower() == "y":
        while True:
            X = collect_input()
            label = predict_single(X, best_model, le)
            print("\n" + "─" * 50)
            print(f"  ★  Predicted Grade Category:  {label}")
            print("─" * 50)
            try:
                again = input("\n  Predict another? [y/N]: ").strip().lower()
            except EOFError:
                again = "n"
            if again != "y":
                break

    banner("Done! All outputs saved to outputs/  and  models/")


if __name__ == "__main__":
    main()
