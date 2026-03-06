# ApexcifyTechnologys — Data Science & Analysis Internship

Submitted as part of the **Data Science & Analysis Internship** at Apexcify Technologys.
This repository contains 3 completed tasks using real-world datasets sourced from Kaggle.

---

## Tasks Completed

### Task 1 — Analyze Student Grades
**Dataset:** Students Performance in Exams — Kaggle (1,000 students)
**File:** `ApexcifyTechnologys_StudentGrades.py`

What the script does:
- Loads the CSV and renames columns for clarity
- Calculates each student's average score across Math, Reading, and Writing
- Assigns letter grades (A–F) based on a standard grading scale
- Identifies the top performer and the student needing most support
- Breaks down performance by gender, test preparation status, and parental education level
- Generates a 6-panel figure: score distributions, grade breakdown, test prep impact,
  gender comparison, parental education correlation, and subject correlation heatmap

Key output: Students who completed test preparation scored **7.6 points higher** on average.
Females outperformed males in Reading and Writing; males led in Math.

---

### Task 2 — Visualize Monthly Sales
**Dataset:** Superstore Sales Dataset — Kaggle (9,994 transactions, 2014–2017)
**File:** `ApexcifyTechnologys_MonthlySales.py`

What the script does:
- Parses Order Date as a proper datetime and extracts year/month
- Aggregates total sales per calendar month for each year independently
- Calculates a 3-month rolling average to surface the underlying trend
- Computes month-over-month percentage growth
- Breaks down revenue by product category, region, and top sub-categories
- Generates a 6-panel figure: yearly trend overlay, rolling average, MoM growth bars,
  category revenue, stacked region + category chart, and top 10 sub-categories

Key output: Revenue grew from $484K (2014) to $733K (2017). November/December
consistently peak every year; February is always the weakest month.

---

### Task 4 — Basic Descriptive Statistics on Dataset
**Dataset:** Titanic — Kaggle (891 passengers, 12 columns)
**File:** `ApexcifyTechnologys_DescriptiveStats.py`

What the script does:
- Loads the real Titanic CSV and engineers additional features (Age Group, Family Size, Log Fare)
- Runs `describe()` for a full statistical summary of all numeric columns
- Reports mean, median, std deviation, min, max, skewness, and percentiles column by column
- Analyses survival rates by passenger class, sex, and age group
- **Bonus:** Detects all missing values using `isnull().sum()` with interpretation
  on how each missing column should be handled in a real pipeline
- Generates a 6-panel figure: age distribution, fare distribution, survival by class,
  age box plots by survival outcome, missing value chart, and feature correlation matrix

Key output: Only 38.4% of passengers survived. 1st class survival rate was 63%
vs 24.2% for 3rd class. Cabin column is 77.1% missing; Age is 19.9% missing.

---

## How to Run

Install dependencies:
```bash
pip install pandas matplotlib seaborn numpy
```

Place the corresponding CSV file in the same folder as each script, then run:

```bash
# Task 1 — needs: StudentsPerformance.csv
python ApexcifyTechnologys_StudentGrades.py

# Task 2 — needs: Sample - Superstore.csv
python ApexcifyTechnologys_MonthlySales.py

# Task 4 — needs: Titanic-Dataset.csv
python ApexcifyTechnologys_DescriptiveStats.py
```

Each script prints a full analysis to the console and saves a chart as a `.png` file.

---

## Datasets

| Task | Dataset | Source |
|------|---------|--------|
| Task 1 | Students Performance in Exams | [Kaggle](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams) |
| Task 2 | Superstore Sales Dataset | [Kaggle](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final) |
| Task 4 | Titanic Dataset | [Kaggle](https://www.kaggle.com/datasets/yasserh/titanic-dataset) |

---

## Tools & Libraries

- Python 3.x
- Pandas — data loading, cleaning, aggregation
- Matplotlib — all charts and visualizations
- Seaborn — heatmaps and statistical plots
- NumPy — numerical operations and feature engineering

---

*Apexcify Technologys Data Science & Analysis Internship*
