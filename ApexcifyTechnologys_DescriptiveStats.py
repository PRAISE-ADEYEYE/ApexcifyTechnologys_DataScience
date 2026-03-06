# Apexcify Technologys — Data Science & Analysis Internship
# Task 4: Basic Descriptive Statistics on Dataset
# Dataset: Titanic (Kaggle, n=891 passengers)
# Tools: Python, Pandas, Matplotlib, Seaborn
# Bonus: Missing value detection with isnull().sum()

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np

# Load the real Titanic dataset — 891 passengers, 12 columns
df = pd.read_csv("Titanic-Dataset.csv")

# Survived is stored as 0/1 — a readable label makes group comparisons self-explanatory
df["Survived_Label"] = df["Survived"].map({1: "Survived", 0: "Did Not Survive"})

# Passenger class is an ordinal integer; a string label prevents it from being
# treated as a continuous variable in any groupby or plot
df["Class_Label"] = df["Pclass"].map({1: "1st Class", 2: "2nd Class", 3: "3rd Class"})

# Age bins let us study survival patterns across life stages rather than individual ages
df["Age_Group"] = pd.cut(df["Age"],
                          bins=[0, 12, 18, 35, 60, 100],
                          labels=["Child (0–12)", "Teen (13–18)",
                                  "Young Adult (19–35)", "Adult (36–60)",
                                  "Senior (60+)"])

# A combined feature captures the intuition that family size affects evacuation outcomes
df["Family_Size"] = df["SibSp"] + df["Parch"] + 1   # +1 includes the passenger themselves

# Fare is right-skewed — log-transformed fare is useful for correlation analysis
df["Log_Fare"] = np.log1p(df["Fare"])   # log1p avoids log(0) errors for zero fares


print("Dataset Overview")
print(f"  Rows (passengers)  : {df.shape[0]}")
print(f"  Columns            : {df.shape[1]}")
print(f"  Columns list       : {', '.join(df.columns.tolist())}")

print("\nFull describe() Output — Numeric Columns")
print(df[["Age", "Fare", "SibSp", "Parch", "Family_Size", "Log_Fare"]]
      .describe().round(2).to_string())

print("\nColumn-by-Column Breakdown")
stat_cols = {"Age": "years", "Fare": "$", "SibSp": "relatives aboard",
             "Parch": "parents/children", "Family_Size": "total incl. self"}

for col, unit in stat_cols.items():
    s = df[col].dropna()
    print(f"\n  {col}  ({unit})")
    print(f"    Count      : {s.count()}")
    print(f"    Mean       : {s.mean():.2f}")
    print(f"    Median     : {s.median():.2f}")
    print(f"    Std Dev    : {s.std():.2f}")
    print(f"    Min        : {s.min():.2f}")
    print(f"    Max        : {s.max():.2f}")
    print(f"    25th pct   : {s.quantile(0.25):.2f}")
    print(f"    75th pct   : {s.quantile(0.75):.2f}")
    print(f"    Skewness   : {s.skew():.2f}")

print("\nSurvival Statistics")
survival_rate = df["Survived"].mean() * 100
print(f"  Overall survival rate     : {survival_rate:.1f}%")
print(f"  Total survivors           : {df['Survived'].sum()}")
print(f"  Total fatalities          : {(df['Survived'] == 0).sum()}")

print("\nSurvival Rate by Passenger Class")
print(df.groupby("Class_Label")["Survived"].mean()
        .apply(lambda x: f"{x*100:.1f}%").to_string())

print("\nSurvival Rate by Sex")
print(df.groupby("Sex")["Survived"].mean()
        .apply(lambda x: f"{x*100:.1f}%").to_string())

print("\nSurvival Rate by Age Group")
print(df.groupby("Age_Group", observed=True)["Survived"].mean()
        .apply(lambda x: f"{x*100:.1f}%").to_string())

print("\nAverage Fare by Passenger Class")
print(df.groupby("Class_Label")["Fare"].mean()
        .apply(lambda x: f"${x:.2f}").to_string())

# BONUS — Missing value analysis using isnull().sum()
# This is a standard first step in any real data pipeline: before doing anything else,
# know exactly what you're missing and how much of the dataset it represents.
print("\nBONUS — Missing Value Analysis  (isnull().sum())")
missing     = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df  = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
missing_df  = missing_df[missing_df["Missing Count"] > 0].sort_values("Missing Count", ascending=False)

if missing_df.empty:
    print("  No missing values found in this dataset.")
else:
    print(missing_df.to_string())
    print("\n  Interpretation:")
    if "Cabin" in missing_df.index:
        print(f"  Cabin   : {missing_df.loc['Cabin','Missing %']}% missing — too sparse to use directly; "
              f"commonly dropped or converted to a 'has cabin record' binary flag.")
    if "Age" in missing_df.index:
        print(f"  Age     : {missing_df.loc['Age','Missing %']}% missing — typically imputed with "
              f"median age or median age within each passenger class.")
    if "Embarked" in missing_df.index:
        print(f"  Embarked: {missing_df.loc['Embarked','Missing %']}% missing — only {missing_df.loc['Embarked','Missing Count']} "
              f"rows; safe to fill with the mode ('{df['Embarked'].mode()[0]}').")


# Visualization — 6-panel figure giving a complete statistical portrait of the dataset
fig = plt.figure(figsize=(20, 12))
fig.suptitle("Titanic Dataset — Descriptive Statistics & Survival Analysis  (n = 891)",
             fontsize=15, fontweight="bold", color="#1a1a2e", y=1.01)
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32)

BLUE   = "#4361ee"
GREEN  = "#2dc653"
AMBER  = "#ffd166"
RED    = "#e63946"
PURPLE = "#7209b7"

# Panel 1 — Age distribution as a histogram with overlaid KDE and vertical lines for mean/median.
# The gap between mean and median reveals the right-skew introduced by the relatively small
# number of elderly passengers pulling the mean upward.
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(df["Age"].dropna(), bins=30, color=BLUE, edgecolor="white",
          linewidth=0.5, alpha=0.75, density=True)
df["Age"].dropna().plot.kde(ax=ax1, color="#1a1a2e", linewidth=2.2)
ax1.axvline(df["Age"].mean(),   color=AMBER, linestyle="--", linewidth=2,
            label=f"Mean: {df['Age'].mean():.1f} yrs")
ax1.axvline(df["Age"].median(), color=GREEN, linestyle="-.", linewidth=2,
            label=f"Median: {df['Age'].median():.1f} yrs")
ax1.set_title("Age Distribution", fontweight="bold", fontsize=11)
ax1.set_xlabel("Age (years)")
ax1.set_ylabel("Density")
ax1.legend(fontsize=9)

# Panel 2 — Fare distribution on a log scale because the raw fare is heavily right-skewed
# (a small number of 1st-class passengers paid far more than everyone else).
# Log scale compresses the tail and makes the full shape visible.
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(df["Fare"].clip(upper=300), bins=35, color=GREEN,
          edgecolor="white", linewidth=0.5, alpha=0.75)
ax2.axvline(df["Fare"].mean(),   color=AMBER, linestyle="--", linewidth=2,
            label=f"Mean: ${df['Fare'].mean():.1f}")
ax2.axvline(df["Fare"].median(), color=RED,   linestyle="-.", linewidth=2,
            label=f"Median: ${df['Fare'].median():.1f}")
ax2.set_title("Fare Distribution  (capped at $300)", fontweight="bold", fontsize=11)
ax2.set_xlabel("Fare ($)")
ax2.set_ylabel("Number of Passengers")
ax2.legend(fontsize=9)

# Panel 3 — Survival rate broken down by passenger class.
# 1st-class passengers had much higher survival rates due to proximity to lifeboats
# and preferential evacuation — this panel quantifies that disparity precisely.
ax3 = fig.add_subplot(gs[0, 2])
class_order   = ["1st Class", "2nd Class", "3rd Class"]
surv_by_class = df.groupby("Class_Label")["Survived"].mean() * 100
surv_by_class = surv_by_class.reindex(class_order)
bars = ax3.bar(class_order, surv_by_class.values,
               color=[GREEN, AMBER, RED], edgecolor="white", linewidth=0.7, width=0.55)
ax3.axhline(survival_rate, color="#374151", linestyle=":",
            linewidth=1.5, label=f"Overall: {survival_rate:.1f}%")
for bar, val in zip(bars, surv_by_class.values):
    ax3.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 1.2,
             f"{val:.1f}%",
             ha="center", fontsize=10, fontweight="bold")
ax3.set_title("Survival Rate by Passenger Class", fontweight="bold", fontsize=11)
ax3.set_ylabel("Survival Rate (%)")
ax3.set_ylim(0, 85)
ax3.legend(fontsize=9)

# Panel 4 — Box plots of age stratified by survival outcome and passenger class.
# Notched box plots let us see whether the age profiles of survivors vs non-survivors
# differ significantly within each class — 3rd-class children are a notable exception.
ax4 = fig.add_subplot(gs[1, 0])
survival_groups = [
    df[(df["Survived"] == 1) & (df["Pclass"] == c)]["Age"].dropna()
    for c in [1, 2, 3]
]
death_groups = [
    df[(df["Survived"] == 0) & (df["Pclass"] == c)]["Age"].dropna()
    for c in [1, 2, 3]
]

positions_surv  = [1, 3, 5]
positions_death = [1.8, 3.8, 5.8]

bp1 = ax4.boxplot(survival_groups, positions=positions_surv, widths=0.6,
                   patch_artist=True, notch=True,
                   boxprops=dict(facecolor=GREEN, alpha=0.65),
                   medianprops=dict(color="white", linewidth=2),
                   whiskerprops=dict(linewidth=1.2),
                   capprops=dict(linewidth=1.2),
                   flierprops=dict(marker="o", markersize=3, alpha=0.3, color=GREEN))

bp2 = ax4.boxplot(death_groups, positions=positions_death, widths=0.6,
                   patch_artist=True, notch=True,
                   boxprops=dict(facecolor=RED, alpha=0.65),
                   medianprops=dict(color="white", linewidth=2),
                   whiskerprops=dict(linewidth=1.2),
                   capprops=dict(linewidth=1.2),
                   flierprops=dict(marker="o", markersize=3, alpha=0.3, color=RED))

ax4.set_xticks([1.4, 3.4, 5.4])
ax4.set_xticklabels(["1st Class", "2nd Class", "3rd Class"], fontsize=10)
ax4.set_title("Age Distribution: Survived vs Did Not Survive", fontweight="bold", fontsize=11)
ax4.set_ylabel("Age (years)")
# Manual legend patches since boxplot doesn't auto-generate them
from matplotlib.patches import Patch
ax4.legend(handles=[Patch(facecolor=GREEN, alpha=0.65, label="Survived"),
                     Patch(facecolor=RED,   alpha=0.65, label="Did Not Survive")],
            fontsize=9)

# Panel 5 — Missing values as a horizontal bar chart ranked by severity.
# Showing percentage rather than raw count makes it immediately clear which columns
# need treatment and which are minor enough to ignore.
ax5 = fig.add_subplot(gs[1, 1])
if not missing_df.empty:
    bar_colors = [RED if p > 50 else AMBER if p > 10 else BLUE
                  for p in missing_df["Missing %"]]
    hbars = ax5.barh(missing_df.index, missing_df["Missing %"],
                      color=bar_colors, edgecolor="white", linewidth=0.6)
    ax5.axvline(50, color="#374151", linestyle="--", linewidth=1,
                label="50% threshold")
    for bar, (count, pct) in zip(hbars, zip(missing_df["Missing Count"],
                                              missing_df["Missing %"])):
        ax5.text(bar.get_width() + 0.5,
                 bar.get_y() + bar.get_height() / 2,
                 f"{count} rows  ({pct}%)",
                 va="center", fontsize=8.5)
    ax5.set_xlim(0, 105)
    ax5.legend(fontsize=9)
else:
    ax5.text(0.5, 0.5, "No missing values", ha="center", va="center",
             fontsize=12, color=GREEN, fontweight="bold")
ax5.set_title("Missing Values by Column  (BONUS)", fontweight="bold", fontsize=11)
ax5.set_xlabel("Missing (%)")

# Panel 6 — Correlation heatmap across all meaningful numeric features.
# Family_Size is a derived feature; its correlation with Fare and survival tells us
# whether travelling in larger groups was a help or a hindrance.
ax6 = fig.add_subplot(gs[1, 2])
corr_cols = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare", "Family_Size", "Log_Fare"]
corr = df[corr_cols].corr()
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask, k=1)] = True   # show lower triangle only

sns.heatmap(corr, ax=ax6, annot=True, fmt=".2f",
            cmap="coolwarm", center=0, vmin=-0.6, vmax=0.6,
            mask=mask, linewidths=0.5, linecolor="white",
            cbar_kws={"shrink": 0.75, "label": "Pearson r"})
ax6.set_title("Feature Correlation Matrix", fontweight="bold", fontsize=11)
ax6.tick_params(axis="x", rotation=40)
ax6.tick_params(axis="y", rotation=0)

plt.savefig("task4_descriptive_stats_output.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nChart saved  →  task4_descriptive_stats_output.png")
