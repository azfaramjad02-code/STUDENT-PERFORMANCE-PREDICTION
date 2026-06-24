# Student Performance Prediction System

A production-style Data Mining project that predicts student academic performance
(**Fail / Pass / Merit / Distinction**) using three classification algorithms.

---

## Project Structure

```
Student_Performance_Project/
│
├── dataset/
│   ├── student_data.csv          ← auto-generated realistic dataset (1 000 rows)
│   └── generate_data.py          ← standalone dataset generator
│
├── src/
│   ├── data_preprocessing.py     ← load, clean, encode, split
│   ├── train_model.py            ← train DT / RF / NB, save best model
│   ├── evaluate.py               ← metrics, confusion matrices
│   ├── predict.py                ← interactive CLI predictor
│   └── visualization.py          ← EDA + comparison charts
│
├── models/
│   ├── best_model.pkl            ← serialised best classifier
│   └── best_model_le.pkl         ← serialised LabelEncoder
│
├── outputs/                      ← all saved PNG charts
│
├── app.py                        ← main entry point
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1 — Create & activate a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### 3 — Generate the dataset (if not present)

```bash
python dataset/generate_data.py
```

### 4 — Run the full pipeline (train + evaluate + save)

```bash
python app.py
```

This will:
- Load and clean the dataset
- Generate EDA charts → `outputs/`
- Train Decision Tree, Random Forest, and Naïve Bayes
- Print classification reports and a comparison table
- Save the best model to `models/best_model.pkl`
- Optionally prompt for a manual prediction

### 5 — Run standalone prediction

```bash
python src/predict.py
```

Enter a student's values at the prompts and get an instant grade prediction.

---

## Models Trained

| Model          | Notes                                          |
|----------------|------------------------------------------------|
| Decision Tree  | `max_depth=8`, balanced class weights          |
| Random Forest  | 200 trees, `max_depth=10`, balanced weights    |
| Naïve Bayes    | `GaussianNB`, fast baseline                   |

The model with the highest test-set accuracy is automatically selected and saved.

---

## Dataset Columns

| Column                  | Type        | Description                      |
|-------------------------|-------------|----------------------------------|
| `attendance_percentage` | float       | 0 – 100 %                        |
| `study_hours_per_week`  | float       | Hours studied per week           |
| `assignment_score`      | float       | Average assignment mark (0–100)  |
| `quiz_score`            | float       | Average quiz mark (0–100)        |
| `midterm_marks`         | float       | Mid-term exam score (0–100)      |
| `previous_gpa`          | float       | Prior semester GPA (0.0–4.0)     |
| `participation_level`   | categorical | Low / Medium / High              |
| `final_result`          | target      | Fail / Pass / Merit / Distinction|

---

## Output Charts (saved to `outputs/`)

| File                        | Description                      |
|-----------------------------|----------------------------------|
| `01_class_distribution.png` | Bar chart of class counts        |
| `02_correlation_heatmap.png`| Numeric feature correlations     |
| `03_feature_distributions.png` | Box plots per class           |
| `04_pairplot.png`           | Pair plot of key features        |
| `05_model_comparison.png`   | Accuracy bar chart               |
| `cm_*.png`                  | Confusion matrix per model       |

---

## Dependencies

```
pandas ≥ 2.0   numpy ≥ 1.24   scikit-learn ≥ 1.3
matplotlib ≥ 3.7   seaborn ≥ 0.12   joblib ≥ 1.3
```
