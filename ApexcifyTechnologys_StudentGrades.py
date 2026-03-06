# Apexcify Technologys — Data Science & Analysis Internship
# Task 1: Analyze Student Grades
# Dataset: Students Performance in Exams (Kaggle, n=1000)
# Tools: Python, Pandas, Matplotlib, Seaborn

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np

# Load the dataset — 1000 students across 8 columns covering demographics and exam scores
df = pd.read_csv("StudentsPerformance.csv")

# Rename to shorter, cleaner column names used throughout the rest of the script
df.rename(columns={
    "gender":                       "Gender",
    "race/ethnicity":               "Group",
    "parental level of education":  "Parent_Education",
    "lunch":                        "Lunch",
    "test preparation course":      "Test_Prep",
    "math score":                   "Math",
    "reading score":                "Reading",
    "writing score":                "Writing"
}, inplace=True)

# Each student sits three exams — their average is the single best summary of overall performance
df["Average"] = df[["Math", "Reading", "Writing"]].mean(axis=1).round(2)

# Map numeric averages onto a standard A–F letter scale for human-readable reporting
def assign_grade(score):
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    elif score >= 60:
        return "D"
    else:
        return "F"

df["Grade"] = df["Average"].apply(assign_grade)

# Cleaner label for test preparation so it reads naturally in charts and print output
df["Prep_Status"] = df["Test_Prep"].map({
    "completed": "Completed Prep",
    "none":      "No Prep"
})

# Identify the single best and worst performers for headline reporting
top_student    = df.loc[df["Average"].idxmax()]
bottom_student = df.loc[df["Average"].idxmin()]

print("Dataset Overview")
print(f"  Total students  : {len(df)}")
print(f"  Subjects        : Math, Reading, Writing")
print(f"  Score range     : {df[['Math','Reading','Writing']].min().min()} – "
      f"{df[['Math','Reading','Writing']].max().max()}")

print("\nClass-wide Descriptive Statistics")
print(df[["Math", "Reading", "Writing", "Average"]].describe().round(2).to_string())

print("\nTop Performer")
print(f"  Average {top_student['Average']}  |  "
      f"Math {top_student['Math']}  "
      f"Reading {top_student['Reading']}  "
      f"Writing {top_student['Writing']}")

print("\nStudent Needing Most Support")
print(f"  Average {bottom_student['Average']}  |  "
      f"Math {bottom_student['Math']}  "
      f"Reading {bottom_student['Reading']}  "
      f"Writing {bottom_student['Writing']}")

print("\nGrade Distribution")
grade_counts = df["Grade"].value_counts().reindex(["A", "B", "C", "D", "F"])
for grade, count in grade_counts.items():
    pct = count / len(df) * 100
    print(f"  Grade {grade}  :  {count:>4} students  ({pct:.1f}%)")

print("\nAverage Score by Test Preparation Status")
print(df.groupby("Prep_Status")[["Math", "Reading", "Writing", "Average"]]
        .mean().round(2).to_string())

print("\nAverage Score by Gender")
print(df.groupby("Gender")[["Math", "Reading", "Writing", "Average"]]
        .mean().round(2).to_string())

print("\nAverage Score by Parental Education Level")
edu_order = [
    "some high school", "high school", "some college",
    "associate's degree", "bachelor's degree", "master's degree"
]
edu_avg = (df.groupby("Parent_Education")["Average"]
             .mean()
             .reindex(edu_order)
             .round(2))
print(edu_avg.to_string())


# Visualization — 6-panel figure that covers every meaningful cut of the data
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Student Performance Analysis — Kaggle Dataset  (n = 1,000)",
             fontsize=15, fontweight="bold", color="#1a1a2e", y=1.01)

subject_colors = ["#4361ee", "#3a0ca3", "#7209b7"]
grade_palette  = {"A": "#2dc653", "B": "#90e0ef", "C": "#ffd166",
                  "D": "#f4a261", "F": "#e63946"}

# Panel 1 — overlapping histograms + KDE curves for all three subjects.
# Stacking them on the same axis instantly shows which subject has the widest spread
# and whether any subject has a notably different central tendency.
for col, color in zip(["Math", "Reading", "Writing"], subject_colors):
    axes[0, 0].hist(df[col], bins=25, alpha=0.35, color=color,
                    edgecolor="white", linewidth=0.4, label=col, density=True)
    df[col].plot.kde(ax=axes[0, 0], color=color, linewidth=2.2)

axes[0, 0].set_title("Score Distribution by Subject", fontweight="bold", fontsize=11)
axes[0, 0].set_xlabel("Score")
axes[0, 0].set_ylabel("Density")
axes[0, 0].legend(fontsize=9)
axes[0, 0].xaxis.set_major_locator(mticker.MultipleLocator(10))

# Panel 2 — grade distribution as individually colour-coded bars.
# Each grade tier gets its own colour so the chart is self-explanatory at a glance.
grade_order  = ["A", "B", "C", "D", "F"]
bars = axes[0, 1].bar(grade_order,
                       grade_counts[grade_order],
                       color=[grade_palette[g] for g in grade_order],
                       edgecolor="white", linewidth=0.8, width=0.58)
for bar, count in zip(bars, grade_counts[grade_order]):
    axes[0, 1].text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 5,
                    f"{count}\n({count / len(df) * 100:.1f}%)",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold")
axes[0, 1].set_title("Grade Distribution (A – F)", fontweight="bold", fontsize=11)
axes[0, 1].set_xlabel("Grade")
axes[0, 1].set_ylabel("Number of Students")
axes[0, 1].set_ylim(0, grade_counts.max() + 65)

# Panel 3 — notched box plots comparing students who did vs did not complete test prep.
# The notch gives a visual confidence interval around the median,
# so we can see at a glance whether the difference is statistically meaningful.
prep_colors = {"No Prep": "#ef233c", "Completed Prep": "#2b9348"}
for i, status in enumerate(["No Prep", "Completed Prep"]):
    subset = df[df["Prep_Status"] == status]["Average"]
    axes[0, 2].boxplot(subset, positions=[i], widths=0.45,
                        patch_artist=True, notch=True,
                        boxprops=dict(facecolor=prep_colors[status], alpha=0.72),
                        medianprops=dict(color="white", linewidth=2.5),
                        whiskerprops=dict(linewidth=1.3),
                        capprops=dict(linewidth=1.3),
                        flierprops=dict(marker="o", markersize=3,
                                        alpha=0.35, color=prep_colors[status]))
    mean_val = subset.mean()
    axes[0, 2].text(i, mean_val + 2, f"μ = {mean_val:.1f}",
                    ha="center", fontsize=9, color="#1a1a2e", fontweight="bold")

axes[0, 2].set_xticks([0, 1])
axes[0, 2].set_xticklabels(["No Prep", "Completed Prep"], fontsize=10)
axes[0, 2].set_title("Effect of Test Preparation on Score",
                      fontweight="bold", fontsize=11)
axes[0, 2].set_ylabel("Average Score")
axes[0, 2].set_ylim(0, 118)

# Panel 4 — grouped bar chart of mean subject scores split by gender.
# Three bars per gender group allow direct subject-by-subject comparison
# and reveal that reading/writing gaps tend to differ from the math gap.
gender_means = df.groupby("Gender")[["Math", "Reading", "Writing"]].mean()
x     = np.arange(len(gender_means.index))
width = 0.25
for j, (subj, color) in enumerate(zip(["Math", "Reading", "Writing"], subject_colors)):
    bars = axes[1, 0].bar(x + j * width, gender_means[subj],
                           width=width, label=subj,
                           color=color, edgecolor="white", linewidth=0.5)
    for bar in bars:
        axes[1, 0].text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.6,
                        f"{bar.get_height():.0f}",
                        ha="center", va="bottom", fontsize=7.5)

axes[1, 0].set_xticks(x + width)
axes[1, 0].set_xticklabels(gender_means.index, fontsize=10)
axes[1, 0].set_title("Mean Subject Score by Gender", fontweight="bold", fontsize=11)
axes[1, 0].set_ylabel("Mean Score")
axes[1, 0].set_ylim(0, 90)
axes[1, 0].legend(fontsize=9)

# Panel 5 — horizontal bar chart showing how parental education correlates with student averages.
# Horizontal layout is intentional — the education labels are too long for vertical bars.
# Bars are ordered from least to most educated, making the upward trend easy to trace.
edu_avg_clean = edu_avg.dropna()
bar_colors    = plt.cm.Blues(np.linspace(0.35, 0.85, len(edu_avg_clean)))
hbars = axes[1, 1].barh(edu_avg_clean.index, edu_avg_clean.values,
                          color=bar_colors, edgecolor="white", linewidth=0.5)
for bar in hbars:
    axes[1, 1].text(bar.get_width() + 0.4,
                    bar.get_y() + bar.get_height() / 2,
                    f"{bar.get_width():.1f}",
                    va="center", fontsize=8.5)
axes[1, 1].set_title("Mean Student Score by Parental Education",
                      fontweight="bold", fontsize=11)
axes[1, 1].set_xlabel("Mean Average Score")
axes[1, 1].set_xlim(0, 82)

# Panel 6 — lower-triangle correlation heatmap across the three subjects and overall average.
# All four are highly correlated (r > 0.8), which tells us a strong student tends to be
# strong across the board — but the exact coefficients tell us where that link is tightest.
corr = df[["Math", "Reading", "Writing", "Average"]].corr()
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask, k=1)] = True  # hide upper triangle to avoid redundancy

sns.heatmap(corr, ax=axes[1, 2], annot=True, fmt=".2f",
            cmap="coolwarm", vmin=0.6, vmax=1,
            mask=mask, linewidths=0.6, linecolor="white",
            cbar_kws={"shrink": 0.8, "label": "Pearson r"})
axes[1, 2].set_title("Subject Score Correlation Matrix", fontweight="bold", fontsize=11)
axes[1, 2].tick_params(axis="x", rotation=30)
axes[1, 2].tick_params(axis="y", rotation=0)

plt.tight_layout()
plt.savefig("task1_student_grades_output.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nChart saved  →  task1_student_grades_output.png")
