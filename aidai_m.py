"""
Анализ оценок курсов — Датасет АУЦА

Источник классификации (обязательный/элективный): Q3
   "Вы берете этот курс, потому что это (отметьте все подходящие варианты)"

    Q3 = 1  → Обязательный для программы/специальности → группа: "required"
    Q3 = 2  → Элективный (по выбору)                   → группа: "elective"

Главный показатель оценки : Q20 — "Общая оценка курса" (1–5)

Показатели для Логистической Регрессии (все 1–5, хорошая заполняемость):
    "Q5":  "Attended classes",
    "Q6":  "Prepared for class",
    "Q7":  "Preparation helped me",
    "Q8":  "Active participant",
    "Q11": "Instruction balance",
    "Q12": "Intellectually stimulating",
    "Q13": "Tools/platforms effective",
    "Q14": "Grading criteria clear",
    "Q23": "Q&A opportunities",
    "Q24": "Welcoming environment",
    "Q25": "Teacher feedback helped",
    "Q26": "Teacher available",
    "Q27": "Overall teacher rating"
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from scipy.stats import mannwhitneyu, ttest_ind
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report

# data load
df = pd.read_excel("cleaned_all_semesters.xlsx")   # измените путь при необходимости

df["course_type"] = df["Q3"].map({1.0: "required", 2.0: "elective"})
df = df[df["course_type"].notna()].copy()

EVAL_COL   = "Q20"
COURSE_COL = "CourseKeyID"
TYPE_COL   = "course_type"

SUB_SCORES = {
    "Q5":  "Attended classes",
    "Q6":  "Prepared for class",
    "Q7":  "Preparation helped me",
    "Q8":  "Active participant",
    "Q11": "Instruction balance",
    "Q12": "Intellectually stimulating",
    "Q13": "Tools/platforms effective",
    "Q14": "Grading criteria clear",
    "Q23": "Q&A opportunities",
    "Q24": "Welcoming environment",
    "Q25": "Teacher feedback helped",
    "Q26": "Teacher available",
    "Q27": "Overall teacher rating",
}
SUB_COLS   = list(SUB_SCORES.keys())
SUB_LABELS = list(SUB_SCORES.values())

palette = {"required": "#4C8EDA", "elective": "#F08B45"}

print("=" * 65)
print("AUCA Course Evaluation Analysis")
print("=" * 65)
print(f"Rows included (Q3 = 1 or 2 only) : {len(df):,}")
print(f"  Required (Q3 = 1)              : {(df[TYPE_COL] == 'required').sum():,}")
print(f"  Elective (Q3 = 2)              : {(df[TYPE_COL] == 'elective').sum():,}")
print(f"  Unique courses                  : {df[COURSE_COL].nunique():,}")
print(f"\nПримечание: Q3=3 (Gen Ed), Q3=4, Q3=5 excluded.")

# part A

print("\n" + "=" * 65)
print("PART A — Within mixed courses")
print("Same course: some students required (Q3=1), others elective (Q3=2)")
print("=" * 65)

type_per_course = df.groupby(COURSE_COL)[TYPE_COL].nunique()
mixed_courses   = type_per_course[type_per_course > 1].index.tolist()
print(f"\nMixed courses found: {len(mixed_courses)}")

results_a = []
for course in mixed_courses:
    sub = df[df[COURSE_COL] == course].dropna(subset=[EVAL_COL])
    req = sub[sub[TYPE_COL] == "required"][EVAL_COL].values
    elec = sub[sub[TYPE_COL] == "elective"][EVAL_COL].values

    # исключаем курсы, где в любой из групп меньше 5 студентов
    if len(req) < 5 or len(elec) < 5:
        continue
    stat, p = mannwhitneyu(req, elec, alternative="two-sided")
    results_a.append({
        "course": course,
        "n_required": len(req),
        "n_elective": len(elec),
        "mean_required": round(req.mean(), 3),
        "mean_elective": round(elec.mean(), 3),
        "U_statistic": round(stat, 2),
        "p_value": round(p, 4),
        "significant": p < 0.05,
    })

results_a_df = pd.DataFrame(results_a)
n_sig = results_a_df["significant"].sum()
n_total = len(results_a_df)

print(f"Tested  : {n_total} courses (≥ 5 students per group)")
print(f"Significant (p < 0.05): {n_sig} / {n_total} courses")
print(f"\nTop 10 most significant courses:")
print(results_a_df.sort_values("p_value").head(10).to_string(index=False))

#box plot
results_a_df["total_n"] = results_a_df["n_required"] + results_a_df["n_elective"]
top12 = results_a_df.nlargest(12, "total_n")["course"].tolist()
plot_df_a = df[df[COURSE_COL].isin(top12)].dropna(subset=[EVAL_COL])

fig, ax = plt.subplots(figsize=(14, 5))
sns.boxplot(
    data=plot_df_a, x=COURSE_COL, y=EVAL_COL, hue=TYPE_COL,
    palette=palette, order=top12, ax=ax,
    linewidth=1.0, fliersize=2, width=0.6
)
# red mark -> significant courses
for i, course in enumerate(top12):
    row = results_a_df[results_a_df["course"] == course]
    if len(row) and row.iloc[0]["significant"]:
        ax.text(i, 5.18, "*", ha="center", fontsize=15,
                color="crimson", fontweight="bold")

ax.set_title(
    "Part A — Overall course rating: required vs elective students\n"
    "12 largest mixed courses",
    fontsize=11
)
ax.set_xlabel("Course ID")
ax.set_ylabel("Overall course rating")
ax.set_ylim(0, 5.55)
# Изменение названий в легенде
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, ['Required', 'Elective'], title="Enrollment type", fontsize=9)
ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.savefig("part_a_boxplot.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nSaved → part_a_boxplot.png")

# part B — required vs elective
course_groups = df.groupby(COURSE_COL)['Q3'].unique()

strictly_req_courses = course_groups[course_groups.apply(lambda x: len(x) == 1 and x[0] == 1)].index
strictly_elec_courses = course_groups[course_groups.apply(lambda x: len(x) == 1 and x[0] == 2)].index

print(f"Found {len(strictly_req_courses)} strictly REQUIRED courses.")
print(f"Found {len(strictly_elec_courses)} strictly ELECTIVE courses.")

df_strict_req = df[df[COURSE_COL].isin(strictly_req_courses)]
df_strict_elec = df[df[COURSE_COL].isin(strictly_elec_courses)]

req_scores = df_strict_req['Q20'].dropna()
elec_scores = df_strict_elec['Q20'].dropna()

import scipy.stats as stats
t_stat, p_val = stats.ttest_ind(req_scores, elec_scores, equal_var=False)

# descriptive statistics
desc = pd.DataFrame({
    "Group": ["Required", "Elective"],
    "N": [len(req_scores), len(elec_scores)],
    "Mean": [req_scores.mean(), elec_scores.mean()],
    "Median": [req_scores.median(), elec_scores.median()],
    "Std Dev": [req_scores.std(), elec_scores.std()],
    "Min": [req_scores.min(), elec_scores.min()],
    "Max": [req_scores.max(), elec_scores.max()],
}).round(3)

print("\nDescriptive Statistics — Q20 (Overall course rating, 1–5):")
print(desc.to_string(index=False))

# t-test
t_stat, p_val = ttest_ind(req_scores, elec_scores, equal_var=False)
sig_b = p_val < 0.05

print(f"\nWelch Independent t-test:")
print(f"  t-statistic = {t_stat:.4f}")
print(f"  p-value   = {p_val:.4f}  →  {'ЗНАЧИМО ✓' if sig_b else 'Не значимо'}")

# bar gchart
means = [req_scores.mean(), elec_scores.mean()]
errors = [req_scores.sem(), elec_scores.sem()]
labels = ["Required\n", "Elective\n"]
colors = [palette["required"], palette["elective"]]

fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(
    labels, means, yerr=errors, color=colors,
    edgecolor="white", width=0.45, capsize=7, linewidth=1.2,
    error_kw={"elinewidth": 1.5, "capthick": 1.5}
)

for bar, mean in zip(bars, means):
    ax.text(
        bar.get_x() + bar.get_width() / 2, mean / 2,
        f"{mean:.2f}", ha="center", va="center",
        fontsize=12, fontweight="bold", color="white"
    )

ax.set_title(
    "Part B — Mean overall course rating\nRequired vs Elective students",
    fontsize=11
)

ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("part_b_barchart.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → part_b_barchart.png")

# logistic regression
print("\n" + "=" * 65)
print("Logistic Regression")
print("Which sub-scores best distinguish required from elective students?")
print("=" * 65)

ml_df = df[df[SUB_COLS + [TYPE_COL]].notna().all(axis=1)].copy()
ml_df["target"] = (ml_df[TYPE_COL] == "elective").astype(int)

X = ml_df[SUB_COLS].values
y = ml_df["target"].values

print(f"\nML dataset : {len(ml_df):,} rows")
print(f"  Required : {(y == 0).sum():,}")
print(f"  Elective : {(y == 1).sum():,}")

# data standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
lr.fit(X_scaled, y)

coef_df = pd.DataFrame({
    "Q"           : SUB_COLS,
    "Description" : SUB_LABELS,
    "Coefficient" : lr.coef_[0].round(4),
    "|Coef|"      : np.abs(lr.coef_[0]).round(4),
    "Direction"   : ["→ elective" if c > 0 else "→ required"
                    for c in lr.coef_[0]],
}).sort_values("|Coef|", ascending=False)

print("\nCoefficients (standardised features):")
print("  Positive → higher score linked to ELECTIVE")
print("  Negative → higher score linked to REQUIRED")
print("  Larger |value| → stronger influence\n")
print(coef_df[["Q", "Description", "Coefficient", "Direction"]].to_string(index=False))

auc = roc_auc_score(y, lr.predict_proba(X_scaled)[:, 1])
print(f"\nAUC = {auc:.4f}")
print("  > 0.65 → sub-scores meaningfully separate the two groups")
print("  ~ 0.50 → sub-scores do not distinguish the groups\n")
print(classification_report(y, lr.predict(X_scaled),
                            target_names=["required", "elective"]))

# bar chart
coef_sorted = coef_df.sort_values("Coefficient")
bar_colors = ["#4C8EDA" if c < 0 else "#F08B45"
              for c in coef_sorted["Coefficient"]]
bar_labels = [f"{q} — {lbl}"
              for q, lbl in zip(coef_sorted["Q"], coef_sorted["Description"])]

fig, ax = plt.subplots(figsize=(9, 7))
bars = ax.barh(bar_labels, coef_sorted["Coefficient"],
               color=bar_colors, edgecolor="white", height=0.6)
ax.axvline(0, color="black", lw=0.9)

for bar in bars:
    w = bar.get_width()
    ax.text(
        w + (0.006 if w >= 0 else -0.006),
        bar.get_y() + bar.get_height() / 2,
        f"{w:+.3f}", va="center",
        ha="left" if w >= 0 else "right", fontsize=8.5
    )

req_patch = mpatches.Patch(color="#4C8EDA", label="→ required (neg)")
elec_patch = mpatches.Patch(color="#F08B45", label="→ elective (pos)")
ax.legend(handles=[req_patch, elec_patch], fontsize=9, loc="lower right")
ax.set_title(
    "Logistic Regression — Sub-score coefficients\n"
    "Which questions best separate required from elective students?",
    fontsize=11
)
ax.set_xlabel("Coefficient (standardised features)")
plt.tight_layout()
plt.savefig("ml_coefficients.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → ml_coefficients.png")
