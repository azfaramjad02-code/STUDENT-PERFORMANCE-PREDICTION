"""
streamlit_app.py
================
Streamlit front-end for the Student Performance Prediction System.

Pages
-----
  🏠 Dashboard        – project overview + dataset snapshot
  📊 Explore Data     – EDA charts rendered inline
  🤖 Train & Evaluate – run training pipeline in-browser, compare models
  🎯 Predict          – form-based single-student grade predictor

Run:
    streamlit run streamlit_app.py
"""

import os
import sys
import io
import warnings
import logging
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import joblib

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

# ── Ensure src/ is importable ─────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data_preprocessing import preprocess_pipeline, load_data, clean_data
from src.train_model        import train_and_select
from src.evaluate           import evaluate_all

# ── Paths ─────────────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
LE_PATH    = MODEL_PATH.replace(".pkl", "_le.pkl")
OUTPUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Palette ───────────────────────────────────────────────────────────────────
CLR = {
    "navy":    "#1B2A4A",
    "teal":    "#0F7B8C",
    "amber":   "#F59E0B",
    "rose":    "#E05252",
    "green":   "#22C55E",
    "muted":   "#94A3B8",
    "surface": "#F0F4F8",
}
CLASS_CLR = {
    "Fail":        CLR["rose"],
    "Pass":        CLR["amber"],
    "Merit":       CLR["teal"],
    "Distinction": CLR["green"],
}

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  /* Sidebar */
  [data-testid="stSidebar"] {{
      background: {CLR['navy']};
  }}
  [data-testid="stSidebar"] * {{ color: #E2E8F0 !important; }}
  [data-testid="stSidebar"] .stRadio label {{ font-size: 0.95rem; }}

  /* Metric cards */
  div[data-testid="metric-container"] {{
      background: {CLR['surface']};
      border-left: 4px solid {CLR['teal']};
      border-radius: 8px;
      padding: 12px 16px;
  }}

  /* Section headers */
  .section-title {{
      font-size: 1.1rem;
      font-weight: 700;
      color: {CLR['navy']};
      border-bottom: 2px solid {CLR['teal']};
      padding-bottom: 4px;
      margin-bottom: 14px;
  }}

  /* Result badge */
  .grade-badge {{
      display: inline-block;
      padding: 10px 28px;
      border-radius: 50px;
      font-size: 1.5rem;
      font-weight: 800;
      letter-spacing: 1px;
  }}

  /* Stbutton tweak */
  div.stButton > button {{
      background: {CLR['teal']};
      color: white;
      border: none;
      border-radius: 6px;
      font-weight: 600;
      padding: 0.5rem 1.4rem;
  }}
  div.stButton > button:hover {{
      background: {CLR['navy']};
  }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers / cached loaders
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def cached_load_df():
    df = load_data()
    return clean_data(df)


@st.cache_resource(show_spinner=False)
def cached_preprocess():
    return preprocess_pipeline()


def load_saved_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(LE_PATH):
        return joblib.load(MODEL_PATH), joblib.load(LE_PATH)
    return None, None


def fig_to_st(fig):
    """Render a matplotlib figure inside Streamlit."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="white")
    buf.seek(0)
    st.image(buf, use_container_width=True)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 Student Performance")
    st.markdown("**Prediction System**")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠  Dashboard", "📊  Explore Data",
         "🤖  Train & Evaluate", "🎯  Predict Grade"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Models: Decision Tree · Random Forest · Naive Bayes")
    st.caption("Target: Fail / Pass / Merit / Distinction")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════
if page == "🏠  Dashboard":
    st.title("🎓 Student Performance Prediction System")
    st.markdown(
        "A complete data mining pipeline — from raw CSV to trained classifier — "
        "wrapped in an interactive dashboard."
    )

    with st.spinner("Loading dataset …"):
        df = cached_load_df()

    # ── KPI cards ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Dataset at a glance</div>',
                unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Students",    f"{len(df):,}")
    c2.metric("Features",          "7")
    c3.metric("Missing (raw)",     "~5 % per column")
    c4.metric("Classes",           "4")

    st.markdown("---")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown('<div class="section-title">Dataset preview</div>',
                    unsafe_allow_html=True)
        st.dataframe(df.head(20), use_container_width=True, height=380)

    with col_right:
        st.markdown('<div class="section-title">Class distribution</div>',
                    unsafe_allow_html=True)
        class_order = ["Fail", "Pass", "Merit", "Distinction"]
        counts = df["final_result"].value_counts().reindex(class_order).dropna()

        fig, ax = plt.subplots(figsize=(5, 3.8))
        colors  = [CLASS_CLR.get(c, CLR["muted"]) for c in counts.index]
        bars    = ax.barh(counts.index, counts.values,
                          color=colors, edgecolor="white", height=0.55)
        ax.bar_label(bars, padding=5, fontsize=10, fontweight="bold")
        ax.set_xlabel("Students", fontsize=9)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(left=False)
        fig.tight_layout()
        fig_to_st(fig)

        st.markdown('<div class="section-title">Descriptive statistics</div>',
                    unsafe_allow_html=True)
        num_cols = ["attendance_percentage","study_hours_per_week",
                    "assignment_score","quiz_score","midterm_marks","previous_gpa"]
        st.dataframe(
            df[num_cols].describe().round(2).T
            [["mean","std","min","50%","max"]]
            .rename(columns={"50%":"median"}),
            use_container_width=True,
        )

    # ── Model status ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-title">Trained model status</div>',
                unsafe_allow_html=True)
    model, le = load_saved_model()
    if model is not None:
        mname = type(model).__name__.replace("Classifier", " Classifier")
        st.success(f"✅  Trained model found: **{mname}**  —  ready for predictions.")
    else:
        st.warning("⚠️  No trained model found. Go to **Train & Evaluate** to build one.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EXPLORE DATA
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📊  Explore Data":
    st.title("📊 Exploratory Data Analysis")

    with st.spinner("Loading data …"):
        df = cached_load_df()

    tab1, tab2, tab3, tab4 = st.tabs([
        "Correlation", "Feature Distributions", "Pair Plot", "Raw Data"
    ])

    # ── Correlation heat-map ─────────────────────────────────────────────────
    with tab1:
        st.subheader("Feature Correlation Heat-map")
        num_cols = ["attendance_percentage","study_hours_per_week",
                    "assignment_score","quiz_score","midterm_marks","previous_gpa"]
        corr = df[num_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        fig, ax = plt.subplots(figsize=(8, 5.5))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                    cmap="coolwarm", center=0, linewidths=0.5,
                    ax=ax, annot_kws={"size": 9})
        ax.set_title("Pearson Correlation — Numeric Features", fontsize=13, pad=10)
        fig.tight_layout()
        fig_to_st(fig)
        st.caption(
            "Midterm marks and assignment score show the strongest correlation "
            "with each other, as expected. GPA has moderate correlation with all scores."
        )

    # ── Box plots ─────────────────────────────────────────────────────────────
    with tab2:
        st.subheader("Feature Distributions by Grade Category")
        class_order   = ["Fail","Pass","Merit","Distinction"]
        present       = [c for c in class_order if c in df["final_result"].unique()]
        palette       = {c: CLASS_CLR.get(c, CLR["muted"]) for c in present}

        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        axes = axes.flatten()
        for i, col in enumerate(num_cols):
            sns.boxplot(data=df, x="final_result", y=col,
                        order=present, palette=palette,
                        ax=axes[i], linewidth=1.1)
            axes[i].set_title(col.replace("_"," ").title(), fontsize=10)
            axes[i].set_xlabel("")
            axes[i].spines[["top","right"]].set_visible(False)
        fig.suptitle("Feature Distributions by Grade Category",
                     fontsize=13, y=1.01, fontweight="bold")
        fig.tight_layout()
        fig_to_st(fig)

    # ── Pair plot ─────────────────────────────────────────────────────────────
    with tab3:
        st.subheader("Pair Plot — Key Features")
        st.caption("Sampled to 400 rows for performance.")
        key   = ["attendance_percentage","midterm_marks","previous_gpa","final_result"]
        sub   = df[key].dropna().copy()
        if len(sub) > 400:
            sub = sub.sample(400, random_state=42)
        present = [c for c in class_order if c in sub["final_result"].unique()]
        palette = {c: CLASS_CLR.get(c, CLR["muted"]) for c in present}
        g = sns.pairplot(sub, hue="final_result", hue_order=present,
                         palette=palette,
                         plot_kws={"alpha": 0.55, "s": 18},
                         diag_kind="kde")
        g.figure.suptitle("Pair Plot — Attendance, Midterm, GPA", y=1.02, fontsize=12)
        buf = io.BytesIO()
        g.figure.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                         facecolor="white")
        buf.seek(0)
        st.image(buf, use_container_width=True)
        plt.close("all")

    # ── Raw data ──────────────────────────────────────────────────────────────
    with tab4:
        st.subheader("Full Dataset")
        search = st.text_input("Filter by final result (e.g. Fail, Pass, Merit, Distinction)")
        view   = df[df["final_result"].str.contains(search, case=False, na=False)] \
                 if search else df
        st.dataframe(view, use_container_width=True, height=480)
        st.caption(f"Showing {len(view):,} of {len(df):,} rows")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — TRAIN & EVALUATE
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🤖  Train & Evaluate":
    st.title("🤖 Train & Evaluate Models")

    st.info(
        "Click **Train Models** to run the full pipeline "
        "(preprocess → train Decision Tree, Random Forest, Naive Bayes → evaluate → save best model)."
    )

    if st.button("🚀  Train Models"):
        progress = st.progress(0, text="Starting pipeline …")

        with st.spinner("Preprocessing data …"):
            X_train, X_test, y_train, y_test, feat_cols, le, df = preprocess_pipeline()
        progress.progress(25, text="Data ready ✔")
        time.sleep(0.3)

        with st.spinner("Training classifiers …"):
            result = train_and_select(X_train, X_test, y_train, y_test)
        progress.progress(65, text="Training complete ✔")
        time.sleep(0.3)

        with st.spinner("Evaluating …"):
            summary = evaluate_all(result["models"], X_test, y_test, le)
        progress.progress(90, text="Evaluation done ✔")
        time.sleep(0.2)

        # Save label encoder alongside model
        joblib.dump(le, LE_PATH)
        progress.progress(100, text="Model saved ✔")
        st.success(
            f"✅  Best model: **{result['best_name']}**  "
            f"({result['results'][result['best_name']]*100:.2f}% accuracy) — saved to models/"
        )

        # ── Accuracy comparison bar chart ──────────────────────────────────
        st.markdown("### Model Accuracy Comparison")
        names = list(result["results"].keys())
        accs  = [result["results"][n] * 100 for n in names]
        bar_clrs = [CLR["teal"], CLR["green"], CLR["amber"]]

        fig, ax = plt.subplots(figsize=(7, 3.5))
        bars = ax.barh(names, accs, color=bar_clrs[:len(names)],
                       edgecolor="white", height=0.45)
        ax.bar_label(bars, fmt="%.2f%%", padding=5,
                     fontsize=11, fontweight="bold")
        ax.set_xlim(0, 112)
        ax.set_xlabel("Accuracy (%)")
        ax.spines[["top","right","left"]].set_visible(False)
        ax.tick_params(left=False)
        fig.tight_layout()
        fig_to_st(fig)

        # ── Per-model confusion matrices ───────────────────────────────────
        st.markdown("### Confusion Matrices")
        cols_cm = st.columns(len(result["models"]))
        for idx, (mname, clf) in enumerate(result["models"].items()):
            from sklearn.metrics import confusion_matrix
            y_pred = clf.predict(X_test)
            cm     = confusion_matrix(y_test, y_pred)
            cnames = list(le.classes_)

            fig, ax = plt.subplots(figsize=(4, 3.5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=cnames, yticklabels=cnames,
                        linewidths=0.4, ax=ax, annot_kws={"size": 9})
            ax.set_title(mname, fontsize=10, fontweight="bold")
            ax.set_xlabel("Predicted", fontsize=8)
            ax.set_ylabel("Actual",    fontsize=8)
            plt.xticks(rotation=30, ha="right", fontsize=7)
            plt.yticks(rotation=0,  fontsize=7)
            fig.tight_layout()
            with cols_cm[idx]:
                fig_to_st(fig)

        # ── Detailed metrics table ─────────────────────────────────────────
        st.markdown("### Metrics Summary")
        rows = []
        for mname, m in sorted(summary.items(),
                                key=lambda x: x[1]["accuracy"], reverse=True):
            rows.append({
                "Model":    mname,
                "Accuracy": f"{m['accuracy']*100:.2f}%",
                "F1-Macro": f"{m['f1_macro']:.4f}",
                "Best ★":   "✅" if mname == result["best_name"] else "",
            })
        st.dataframe(pd.DataFrame(rows).set_index("Model"),
                     use_container_width=True)

        # ── Classification reports ─────────────────────────────────────────
        from sklearn.metrics import classification_report
        with st.expander("📄  Full classification reports"):
            for mname, clf in result["models"].items():
                y_pred = clf.predict(X_test)
                report = classification_report(
                    y_test, y_pred, target_names=list(le.classes_), zero_division=0
                )
                st.markdown(f"**{mname}**")
                st.code(report, language="text")

    else:
        model, le = load_saved_model()
        if model is not None:
            mname = type(model).__name__
            st.markdown(f"#### Previously trained model found: `{mname}`")
            st.markdown("Hit **Train Models** to retrain with the current dataset.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — PREDICT
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🎯  Predict Grade":
    st.title("🎯 Predict Student Grade")

    model, le = load_saved_model()
    if model is None:
        st.error("No trained model found. Please go to **Train & Evaluate** first.")
        st.stop()

    mname = type(model).__name__
    st.success(f"Using model: **{mname}**")
    st.markdown("Fill in the student's details and click **Predict**.")

    st.markdown("---")

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            attendance = st.slider(
                "Attendance %", 0.0, 100.0, 75.0, 0.5,
                help="Percentage of classes attended"
            )
            study_hours = st.slider(
                "Study hours / week", 1.0, 40.0, 12.0, 0.5,
                help="Average weekly self-study hours"
            )

        with c2:
            assignment = st.slider(
                "Assignment score", 0.0, 100.0, 68.0, 0.5,
                help="Average assignment mark out of 100"
            )
            quiz = st.slider(
                "Quiz score", 0.0, 100.0, 65.0, 0.5,
                help="Average quiz mark out of 100"
            )

        with c3:
            midterm = st.slider(
                "Midterm marks", 0.0, 100.0, 60.0, 0.5,
                help="Mid-term exam score out of 100"
            )
            prev_gpa = st.slider(
                "Previous GPA", 0.0, 4.0, 2.8, 0.05,
                help="GPA from the previous semester"
            )

        participation = st.select_slider(
            "Participation level",
            options=["Low", "Medium", "High"],
            value="Medium",
        )

        submitted = st.form_submit_button("🔮  Predict Grade", use_container_width=True)

    if submitted:
        part_map = {"Low": 0, "Medium": 1, "High": 2}
        X = np.array([[
            attendance, study_hours, assignment, quiz,
            midterm, prev_gpa, part_map[participation]
        ]], dtype=np.float32)

        y_pred = model.predict(X)[0]
        label  = le.inverse_transform([y_pred])[0]

        badge_clr = CLASS_CLR.get(label, CLR["teal"])

        st.markdown("---")
        st.markdown("### Prediction Result")

        r1, r2 = st.columns([1, 2])
        with r1:
            st.markdown(
                f'<div class="grade-badge" '
                f'style="background:{badge_clr}22; color:{badge_clr}; '
                f'border: 3px solid {badge_clr};">'
                f'{label}</div>',
                unsafe_allow_html=True,
            )

        with r2:
            desc = {
                "Distinction": "🌟 Outstanding performance across all areas.",
                "Merit":        "✅ Strong performance — above average results.",
                "Pass":         "👍 Meets requirements — room to grow.",
                "Fail":         "⚠️  At risk of failing — consider extra support.",
            }
            st.markdown(f"**{desc.get(label, '')}**")

            # Input summary
            input_df = pd.DataFrame({
                "Feature": ["Attendance %", "Study hrs/wk", "Assignment",
                            "Quiz", "Midterm", "Prev GPA", "Participation"],
                "Value":   [attendance, study_hours, assignment,
                            quiz, midterm, prev_gpa, participation],
            })
            st.dataframe(input_df.set_index("Feature").T,
                         use_container_width=True)

        # ── Probability bar (if model supports it) ─────────────────────────
        if hasattr(model, "predict_proba"):
            proba  = model.predict_proba(X)[0]
            labels = le.inverse_transform(np.arange(len(proba)))
            order  = ["Fail","Pass","Merit","Distinction"]
            order_idx = [list(labels).index(c) for c in order if c in labels]

            fig, ax = plt.subplots(figsize=(7, 2.8))
            bar_clrs = [CLASS_CLR.get(labels[i], CLR["muted"]) for i in order_idx]
            vals     = [proba[i] for i in order_idx]
            lbls     = [labels[i] for i in order_idx]
            bars = ax.barh(lbls, [v*100 for v in vals],
                           color=bar_clrs, edgecolor="white", height=0.45)
            ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=10, fontweight="bold")
            ax.set_xlim(0, 115)
            ax.set_xlabel("Probability (%)", fontsize=9)
            ax.set_title("Prediction Confidence", fontsize=11, fontweight="bold")
            ax.spines[["top","right","left"]].set_visible(False)
            ax.tick_params(left=False)
            fig.tight_layout()
            st.markdown("#### Confidence per class")
            fig_to_st(fig)
