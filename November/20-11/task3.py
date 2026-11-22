import pandas as pd
import numpy as np

data = pd.read_csv("womens_clothing_segments.csv")

X = data.select_dtypes(include = "number")
X = X.drop(columns = [c for c in X.columns
    if "id" in c.lower() or c.lower() in ["segment", "anomaly"]], errors = "ignore")

report = X.describe().T
report["Variance"] = X.var()
report["Missing"] = X.isnull().sum()

print("\nSummary Statistics:")
print(report.round(2))

insights = []

for col in X.columns:
    mean = X[col].mean()
    median = X[col].median()
    variance = X[col].var()
    skew = X[col].skew()

    if mean > median:
        trend = "has a right-skewed distribution."
    elif mean < median:
        trend = "has a left-skewed distribution."
    else:
        trend = "has a relatively balanced distribution."

    if skew > 1:
        shape = "It has strong positive skewness."
    elif skew < -1:
        shape = "It has strong negative skewness."
    else:
        shape = "Its skewness is relatively low."

    insights.append(
        f"{col}: Mean is {mean:.2f}, median is {median:.2f}, "
        f"and variance is {variance:.2f}. {trend} {shape}"
    )

if "productbrand" in data.columns:
    brand = data["productbrand"].value_counts().head(5)
    insights.append("Top product brands are: " + ", ".join(brand.index.astype(str)))

if "primarycolor" in data.columns:
    color = data["primarycolor"].value_counts().head(5)
    insights.append("Popular product colours are: " + ", ".join(color.index.astype(str)))

if "Segment" in data.columns:
    segments = data["Segment"].value_counts()
    insights.append(
        f"The dataset contains {len(segments)} product segments. "
        f"Largest segment is {segments.idxmax()} "
        f"with {segments.max()} products."
    )

summary = "\n".join(insights)

final_report = f"""
WOMEN'S CLOTHING - FINAL DATA SUMMARY REPORT

Total Products: {len(data)}
Total Features: {len(data.columns)}

SUMMARY STATISTICS:
{report.round(2).to_string()}

AUTOMATED DATA INSIGHTS:
{summary}
"""

print("\n", final_report)

report.to_csv("summary_statistics.csv")
with open("final_summary_report.txt", "w", encoding = "utf-8") as file:
    file.write(final_report)

print("\nReports generated successfully!")