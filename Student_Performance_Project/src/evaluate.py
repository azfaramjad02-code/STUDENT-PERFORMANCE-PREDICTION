"""
evaluate.py
===========
Prints full classification reports, draws confusion matrices for every
trained model, and saves them to outputs/.
"""

import os
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder

log = logging.getLogger(__name__)

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_ORDER = ["Distinction", "Fail", "Merit", "Pass"]   # alphabetical (sklearn default)


def evaluate_all(
    models:   dict,
    X_test:   np.ndarray,
    y_test:   np.ndarray,
    le:       LabelEncoder,
) -> dict:
    """
    For each model:
      1. Print accuracy + classification report
      2. Save confusion matrix to outputs/
    Returns {model_name: {"accuracy": float, "f1_macro": float}}
    """
    class_names = list(le.classes_)
    summary = {}

    for name, clf in models.items():
        y_pred = clf.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        f1     = f1_score(y_test, y_pred, average="macro", zero_division=0)
        summary[name] = {"accuracy": acc, "f1_macro": f1}

        print(f"\n{'═'*55}")
        print(f"  Model : {name}")
        print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f} %)")
        print(f"  F1 Macro  : {f1:.4f}")
        print(f"{'─'*55}")
        print(classification_report(
            y_test, y_pred,
            target_names=class_names,
            zero_division=0,
        ))

        _plot_confusion_matrix(y_test, y_pred, class_names, name)

    return summary


def _plot_confusion_matrix(y_true, y_pred, class_names, model_name):
    cm   = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual",    fontsize=11)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=12, pad=10)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()

    safe = model_name.lower().replace(" ", "_")
    path = os.path.join(OUTPUT_DIR, f"cm_{safe}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Confusion matrix saved → {path}")


def print_summary_table(summary: dict):
    """Pretty-print a comparison table."""
    print(f"\n{'═'*45}")
    print(f"  {'Model':<22} {'Accuracy':>10}  {'F1-Macro':>10}")
    print(f"{'─'*45}")
    for name, metrics in sorted(summary.items(),
                                 key=lambda x: x[1]["accuracy"],
                                 reverse=True):
        print(f"  {name:<22} {metrics['accuracy']:>9.4f}  {metrics['f1_macro']:>10.4f}")
    print(f"{'═'*45}")
