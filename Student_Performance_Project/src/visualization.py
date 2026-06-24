"""
visualization.py
================
Exploratory Data Analysis charts:
  • Class distribution
  • Correlation heat-map
  • Feature distributions by class
  • Pair plot of key numeric features
All figures are saved to outputs/ and displayed on screen.
"""

import os
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for all envs)
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUMERIC_COLS = [
    "attendance_percentage",
    "study_hours_per_week",
    "assignment_score",
    "quiz_score",
    "midterm_marks",
    "previous_gpa",
]
TARGET = "final_result"

# Palette: one colour per class for consistency
CLASS_ORDER  = ["Fail", "Pass", "Merit", "Distinction"]
CLASS_COLORS = ["#e74c3c", "#f39c12", "#2ecc71", "#3498db"]


def _save(fig, name: str):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Chart saved → {path}")


def plot_class_distribution(df: pd.DataFrame):
    """Bar chart of target class counts."""
    counts = df[TARGET].value_counts().reindex(CLASS_ORDER).dropna()
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(counts.index, counts.values,
                  color=CLASS_COLORS[:len(counts)], edgecolor="white", width=0.6)
    ax.bar_label(bars, padding=4, fontsize=11, fontweight="bold")
    ax.set_title("Student Performance — Class Distribution", fontsize=14, pad=12)
    ax.set_xlabel("Final Result", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, "01_class_distribution.png")


def plot_correlation_heatmap(df: pd.DataFrame):
    """Heat-map of numeric feature correlations."""
    corr = df[NUMERIC_COLS].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="coolwarm", center=0, linewidths=0.5,
        ax=ax, annot_kws={"size": 9},
    )
    ax.set_title("Feature Correlation Heat-map", fontsize=14, pad=12)
    fig.tight_layout()
    _save(fig, "02_correlation_heatmap.png")


def plot_feature_distributions(df: pd.DataFrame):
    """Box plots of each numeric feature grouped by class."""
    n = len(NUMERIC_COLS)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    present_classes = [c for c in CLASS_ORDER if c in df[TARGET].unique()]
    palette = {c: col for c, col in zip(CLASS_ORDER, CLASS_COLORS) if c in present_classes}

    for i, col in enumerate(NUMERIC_COLS):
        sns.boxplot(
            data=df, x=TARGET, y=col,
            order=present_classes, palette=palette,
            ax=axes[i], linewidth=1.2,
        )
        axes[i].set_title(col.replace("_", " ").title(), fontsize=10)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")
        axes[i].spines[["top", "right"]].set_visible(False)

    fig.suptitle("Feature Distributions by Class", fontsize=14, y=1.01)
    fig.tight_layout()
    _save(fig, "03_feature_distributions.png")


def plot_pairplot(df: pd.DataFrame):
    """Seaborn pair plot for a subset of key features."""
    key_features = ["attendance_percentage", "midterm_marks", "previous_gpa", TARGET]
    subset = df[key_features].dropna().copy()
    # Limit to 400 rows for speed
    if len(subset) > 400:
        subset = subset.sample(400, random_state=42)

    present = [c for c in CLASS_ORDER if c in subset[TARGET].unique()]
    palette  = {c: col for c, col in zip(CLASS_ORDER, CLASS_COLORS) if c in present}

    g = sns.pairplot(
        subset, hue=TARGET, hue_order=present,
        palette=palette, plot_kws={"alpha": 0.6, "s": 20},
        diag_kind="kde",
    )
    g.figure.suptitle("Pair Plot — Key Features", y=1.02, fontsize=13)
    _save(g.figure, "04_pairplot.png")


def plot_model_comparison(results: dict):
    """
    Bar chart comparing model accuracies.
    results = {"ModelName": accuracy_float, ...}
    """
    names = list(results.keys())
    accs  = [results[n] * 100 for n in names]
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(names, accs, color=colors[:len(names)], edgecolor="white", height=0.5)
    ax.bar_label(bars, fmt="%.2f%%", padding=4, fontsize=11, fontweight="bold")
    ax.set_xlim(0, 110)
    ax.set_xlabel("Accuracy (%)", fontsize=11)
    ax.set_title("Model Accuracy Comparison", fontsize=14, pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, "05_model_comparison.png")
    return os.path.join(OUTPUT_DIR, "05_model_comparison.png")


def run_eda(df: pd.DataFrame):
    """Run all EDA charts in sequence."""
    log.info("Running EDA visualisations …")
    plot_class_distribution(df)
    plot_correlation_heatmap(df)
    plot_feature_distributions(df)
    plot_pairplot(df)
    log.info("All EDA charts saved to outputs/")
