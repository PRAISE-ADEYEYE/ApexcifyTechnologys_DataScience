# Apexcify Technologys — Data Science & Analysis Internship
# Task 2: Visualize Monthly Sales
# Dataset: Superstore Sales (Kaggle, n=9,994 transactions)
# Tools: Python, Pandas, Matplotlib

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# The Superstore CSV uses Windows-1252 encoding — latin1 handles it without errors
df = pd.read_csv("Sample - Superstore.csv", encoding="latin1")

# Parse Order Date as a proper datetime object so we can extract year and month cleanly
df["Order Date"] = pd.to_datetime(df["Order Date"], format="%m/%d/%Y")
df["Year"]       = df["Order Date"].dt.year
df["Month"]      = df["Order Date"].dt.month
df["Month_Name"] = df["Order Date"].dt.strftime("%b")   # Jan, Feb, ...
df["YearMonth"]  = df["Order Date"].dt.to_period("M")   # 2017-01, 2017-02, ...

# The dataset spans multiple years — we analyse each year individually
# so monthly trends are not averaged across calendar years, which would hide seasonality
years_available = sorted(df["Year"].unique())
print(f"Dataset Overview")
print(f"  Total transactions  : {len(df):,}")
print(f"  Date range          : {df['Order Date'].min().date()} – {df['Order Date'].max().date()}")
print(f"  Years in dataset    : {years_available}")
print(f"  Total revenue       : ${df['Sales'].sum():,.2f}")
print(f"  Categories          : {', '.join(df['Category'].unique())}")

# Aggregate total sales per calendar month for each year
monthly = (df.groupby(["Year", "Month", "Month_Name"])["Sales"]
             .sum()
             .reset_index()
             .sort_values(["Year", "Month"]))

# A 3-month rolling average smooths out noise and reveals the true underlying trend.
# min_periods=1 ensures January still gets a value even though there are no prior months.
monthly["Rolling_3M"] = (monthly.groupby("Year")["Sales"]
                                  .transform(lambda x: x.rolling(3, min_periods=1).mean()))

# Month-over-month percentage change tells us whether growth is accelerating or slowing
monthly["MoM_Growth"] = (monthly.groupby("Year")["Sales"]
                                  .transform(lambda x: x.pct_change() * 100))

print("\nMonthly Sales Table")
print(monthly[["Year", "Month_Name", "Sales", "Rolling_3M", "MoM_Growth"]]
      .rename(columns={"Month_Name": "Month", "Rolling_3M": "3M Avg",
                        "MoM_Growth": "MoM %"})
      .to_string(index=False, float_format=lambda x: f"{x:,.1f}"))

for year in years_available:
    subset = monthly[monthly["Year"] == year]
    best   = subset.loc[subset["Sales"].idxmax()]
    worst  = subset.loc[subset["Sales"].idxmin()]
    print(f"\n{year} Summary")
    print(f"  Total revenue   : ${subset['Sales'].sum():,.2f}")
    print(f"  Monthly average : ${subset['Sales'].mean():,.2f}")
    print(f"  Best month      : {best['Month_Name']}  —  ${best['Sales']:,.2f}")
    print(f"  Weakest month   : {worst['Month_Name']}  —  ${worst['Sales']:,.2f}")
    print(f"  Peak MoM growth : {subset['MoM_Growth'].max():.1f}%")

print("\nSales by Product Category")
print(df.groupby("Category")["Sales"].sum().sort_values(ascending=False)
        .apply(lambda x: f"${x:,.2f}").to_string())

print("\nSales by Region")
print(df.groupby("Region")["Sales"].sum().sort_values(ascending=False)
        .apply(lambda x: f"${x:,.2f}").to_string())

print("\nTop 5 Sub-Categories by Revenue")
print(df.groupby("Sub-Category")["Sales"].sum().sort_values(ascending=False)
        .head(5).apply(lambda x: f"${x:,.2f}").to_string())


# Visualization — 6-panel layout covering trend, growth, category, region, and segment
fig, axes = plt.subplots(2, 3, figsize=(20, 11))
fig.suptitle("Superstore Monthly Sales Analysis — Kaggle Dataset",
             fontsize=15, fontweight="bold", color="#1a1a2e", y=1.01)

year_colors  = ["#4361ee", "#e63946", "#2dc653", "#ff9f1c"]
region_cols  = ["#4361ee", "#e63946", "#2dc653", "#ff9f1c"]

# Panel 1 — line chart of monthly sales per year with a shaded area under the most recent year.
# Overlaying years on the same x-axis (Jan–Dec) reveals whether seasonality is consistent
# or whether one year had an unusual spike relative to the others.
month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]

for idx, year in enumerate(years_available):
    subset = monthly[monthly["Year"] == year].sort_values("Month")
    color  = year_colors[idx % len(year_colors)]
    axes[0, 0].plot(subset["Month"], subset["Sales"],
                    marker="o", linewidth=2.3, markersize=6,
                    color=color, label=str(year))
    if year == years_available[-1]:
        axes[0, 0].fill_between(subset["Month"], subset["Sales"],
                                 alpha=0.10, color=color)

axes[0, 0].set_xticks(range(1, 13))
axes[0, 0].set_xticklabels(month_labels, fontsize=9)
axes[0, 0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
axes[0, 0].set_title("Monthly Sales Trend by Year", fontweight="bold", fontsize=11)
axes[0, 0].set_xlabel("Month")
axes[0, 0].set_ylabel("Total Sales")
axes[0, 0].legend(fontsize=9)

# Panel 2 — 3-month rolling average per year.
# Stripping out the bar-by-bar volatility surfaces the medium-term direction more cleanly
# than raw monthly values can — useful when making forward-looking decisions.
for idx, year in enumerate(years_available):
    subset = monthly[monthly["Year"] == year].sort_values("Month")
    color  = year_colors[idx % len(year_colors)]
    axes[0, 1].plot(subset["Month"], subset["Rolling_3M"],
                    linewidth=2.5, color=color, label=str(year))
    axes[0, 1].scatter(subset["Month"], subset["Rolling_3M"],
                        s=35, color=color, zorder=3)

axes[0, 1].set_xticks(range(1, 13))
axes[0, 1].set_xticklabels(month_labels, fontsize=9)
axes[0, 1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
axes[0, 1].set_title("3-Month Rolling Average Sales", fontweight="bold", fontsize=11)
axes[0, 1].set_xlabel("Month")
axes[0, 1].set_ylabel("Rolling Average Sales")
axes[0, 1].legend(fontsize=9)

# Panel 3 — month-over-month growth bars coloured green (growth) or red (decline).
# This makes it immediately obvious which months are acceleration points
# and which are seasonal pullbacks.
latest_year = years_available[-1]
mom_data    = monthly[monthly["Year"] == latest_year].sort_values("Month")
mom_vals    = mom_data["MoM_Growth"].fillna(0).values
bar_colors  = ["#2dc653" if v >= 0 else "#e63946" for v in mom_vals]

bars = axes[0, 2].bar(mom_data["Month_Name"], mom_vals,
                       color=bar_colors, edgecolor="white", linewidth=0.7, width=0.65)
axes[0, 2].axhline(0, color="#374151", linewidth=0.9)
for bar, val in zip(bars, mom_vals):
    if abs(val) > 0.5:
        axes[0, 2].text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (1.2 if val >= 0 else -3.5),
                        f"{val:.1f}%",
                        ha="center", fontsize=7.5,
                        color="#1a1a2e", fontweight="bold")
axes[0, 2].set_title(f"Month-over-Month Sales Growth ({latest_year})",
                      fontweight="bold", fontsize=11)
axes[0, 2].set_xlabel("Month")
axes[0, 2].set_ylabel("MoM Change (%)")
axes[0, 2].tick_params(axis="x", rotation=30)

# Panel 4 — horizontal bar chart of revenue by product category.
# Horizontal bars are cleaner here because the category names are read more
# comfortably left-to-right than rotated beneath vertical bars.
cat_sales = (df.groupby("Category")["Sales"].sum()
               .sort_values(ascending=True))   # ascending so largest bar is on top
cat_colors = ["#4361ee", "#7209b7", "#3a0ca3"]
hbars = axes[1, 0].barh(cat_sales.index, cat_sales.values,
                          color=cat_colors, edgecolor="white", linewidth=0.6)
for bar in hbars:
    axes[1, 0].text(bar.get_width() + 1500,
                    bar.get_y() + bar.get_height() / 2,
                    f"${bar.get_width():,.0f}",
                    va="center", fontsize=9, fontweight="bold")
axes[1, 0].set_title("Total Revenue by Product Category",
                      fontweight="bold", fontsize=11)
axes[1, 0].set_xlabel("Total Sales ($)")
axes[1, 0].xaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))

# Panel 5 — stacked bar chart of sales broken down by both region and product category.
# Stacking lets us see the total per region while also seeing how the category mix differs —
# for example, Technology may dominate the West while Furniture leads in the South.
region_cat = (df.groupby(["Region", "Category"])["Sales"]
                .sum()
                .unstack(fill_value=0))
region_cat = region_cat.loc[region_cat.sum(axis=1).sort_values(ascending=False).index]

cat_stack_colors = ["#4361ee", "#7209b7", "#e63946"]
bottom = np.zeros(len(region_cat))
for cat, color in zip(region_cat.columns, cat_stack_colors):
    bars = axes[1, 1].bar(region_cat.index, region_cat[cat],
                           bottom=bottom, label=cat,
                           color=color, edgecolor="white", linewidth=0.5, width=0.55)
    bottom += region_cat[cat].values

axes[1, 1].yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
axes[1, 1].set_title("Sales by Region & Category (Stacked)",
                      fontweight="bold", fontsize=11)
axes[1, 1].set_xlabel("Region")
axes[1, 1].set_ylabel("Total Sales ($)")
axes[1, 1].legend(title="Category", fontsize=8, title_fontsize=9)
axes[1, 1].tick_params(axis="x", rotation=15)

# Panel 6 — top 10 sub-categories ranked by revenue as a horizontal bar chart.
# This is the most granular product-level view and often surfaces surprises —
# Phones and Chairs typically outsell entire minor categories.
sub_sales = (df.groupby("Sub-Category")["Sales"].sum()
               .sort_values(ascending=True)
               .tail(10))
gradient  = plt.cm.viridis(np.linspace(0.25, 0.85, len(sub_sales)))
hbars2 = axes[1, 2].barh(sub_sales.index, sub_sales.values,
                           color=gradient, edgecolor="white", linewidth=0.5)
for bar in hbars2:
    axes[1, 2].text(bar.get_width() + 500,
                    bar.get_y() + bar.get_height() / 2,
                    f"${bar.get_width():,.0f}",
                    va="center", fontsize=8)
axes[1, 2].set_title("Top 10 Sub-Categories by Revenue",
                      fontweight="bold", fontsize=11)
axes[1, 2].set_xlabel("Total Sales ($)")
axes[1, 2].xaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))

plt.tight_layout()
plt.savefig("task2_monthly_sales_output.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nChart saved  →  task2_monthly_sales_output.png")
