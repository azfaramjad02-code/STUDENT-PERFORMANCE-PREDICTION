"""
train_model.py
==============
Trains three classifiers:
  • Decision Tree
  • Random Forest
  • Gaussian Naïve Bayes

Selects the best by test-set accuracy, serialises it to models/best_model.pkl,
and returns a results dict consumed by evaluate.py and app.py.
"""

import os
import logging
import joblib
import numpy as np
from sklearn.tree          import DecisionTreeClassifier
from sklearn.ensemble      import RandomForestClassifier
from sklearn.naive_bayes   import GaussianNB
from sklearn.metrics       import accuracy_score

log = logging.getLogger(__name__)

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
os.makedirs(MODELS_DIR, exist_ok=True)


def build_classifiers() -> dict:
    """Return a dict of {name: unfitted_estimator}."""
    return {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            min_samples_split=10,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
        "Naive Bayes": GaussianNB(var_smoothing=1e-9),
    }


def train_and_select(
    X_train: np.ndarray,
    X_test:  np.ndarray,
    y_train: np.ndarray,
    y_test:  np.ndarray,
) -> dict:
    """
    Train every classifier, evaluate on the test split, pick the winner.

    Returns
    -------
    dict with keys:
        results   – {model_name: accuracy}
        best_name – str
        best_model– fitted estimator
        models    – {model_name: fitted_estimator}
    """
    classifiers = build_classifiers()
    results  = {}
    trained  = {}

    for name, clf in classifiers.items():
        log.info(f"Training  [{name}] …")
        try:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc    = accuracy_score(y_test, y_pred)
            results[name] = acc
            trained[name] = clf
            log.info(f"  └─ Accuracy: {acc:.4f}")
        except Exception as exc:
            log.error(f"  └─ Training failed: {exc}")

    if not results:
        raise RuntimeError("All models failed to train.")

    # ── Select best ──────────────────────────────────────────────────────────
    best_name  = max(results, key=results.get)
    best_model = trained[best_name]
    best_acc   = results[best_name]
    log.info(f"\n★  Best model: {best_name}  (accuracy={best_acc:.4f})")

    # ── Persist ───────────────────────────────────────────────────────────────
    joblib.dump(best_model, MODEL_PATH)
    log.info(f"Model saved → {MODEL_PATH}")

    return {
        "results":    results,
        "best_name":  best_name,
        "best_model": best_model,
        "models":     trained,
    }


def load_best_model():
    """Load the serialised best model from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}. "
            "Run train_model.py (or app.py) first."
        )
    return joblib.load(MODEL_PATH)
