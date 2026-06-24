"""
Script to generate a realistic student performance dataset.
Run this once to create student_data.csv in the dataset/ folder.
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)
N = 1000

attendance      = np.clip(np.random.normal(75, 15, N), 20, 100)
study_hours     = np.clip(np.random.normal(12, 5, N), 1, 40)
assignment      = np.clip(np.random.normal(68, 15, N), 20, 100)
quiz            = np.clip(np.random.normal(65, 15, N), 10, 100)
midterm         = np.clip(np.random.normal(60, 18, N), 10, 100)
prev_gpa        = np.clip(np.random.normal(2.8, 0.7, N), 0.0, 4.0)
participation   = np.random.choice(["Low", "Medium", "High"], N, p=[0.25, 0.45, 0.30])

# Deterministic score → final label (with a little noise)
score = (
    0.20 * attendance +
    0.15 * study_hours * 2.5 +
    0.20 * assignment +
    0.15 * quiz +
    0.20 * midterm +
    0.10 * prev_gpa * 25
)
participation_bonus = {"Low": -5, "Medium": 0, "High": 5}
score += np.array([participation_bonus[p] for p in participation])
score += np.random.normal(0, 5, N)

# Missing values (~5 %)
for col_arr in [attendance, study_hours, assignment, quiz, midterm, prev_gpa]:
    idx = np.random.choice(N, size=int(0.05 * N), replace=False)
    col_arr[idx] = np.nan

final_result = pd.cut(
    score,
    bins=[-np.inf, 50, 65, 80, np.inf],
    labels=["Fail", "Pass", "Merit", "Distinction"]
)

df = pd.DataFrame({
    "attendance_percentage": np.round(attendance, 1),
    "study_hours_per_week":  np.round(study_hours, 1),
    "assignment_score":      np.round(assignment, 1),
    "quiz_score":            np.round(quiz, 1),
    "midterm_marks":         np.round(midterm, 1),
    "previous_gpa":          np.round(prev_gpa, 2),
    "participation_level":   participation,
    "final_result":          final_result,
})

out = os.path.join(os.path.dirname(__file__), "student_data.csv")
df.to_csv(out, index=False)
print(f"Dataset saved → {out}  ({len(df)} rows)")
