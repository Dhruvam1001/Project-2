import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("womens_clothing_scaled.csv")
cols = [c for c in data.select_dtypes(include = "number").columns
        if "id" not in c.lower() and c.lower() != "target"]

def histogram(s, col):
    s.hist(bins = 20, edgecolor = "black")
    plt.title(col + " - Histogram")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()

def boxplot(s, col):
    plt.boxplot(s)
    plt.title(col + " - Boxplot")
    plt.ylabel(col)
    plt.show()

def kde_plot(s, col):
    s = s.dropna()

    print("\nAnalyzing:", col)

    if s.nunique() < 2:
        print("KDE cannot be generated: insufficient variation.")
        return

    try:
        s.plot(kind = "kde")
        plt.title(f"KDE Plot - {col}")
        plt.show()

    except np.linalg.LinAlgError:
        print("KDE cannot be generated for this column.")

results = []

for col in cols:
    s = data[col].dropna()

    if len(s) < 2:
        continue

    skew = s.skew()
    kurt = s.kurt()

    if skew > 1:
        suggestion = "Log transformation"
    elif skew < -1:
        suggestion = "Power transformation"
    else:
        suggestion = "No major skewness"

    results.append([col, round(skew, 3), round(kurt, 3), suggestion])

    print("\nAnalyzing:", col)
    histogram(s, col)
    boxplot(s, col)
    kde_plot(s, col)

report = pd.DataFrame(results, columns = ["Feature", "Skewness", "Kurtosis", "Suggested Transformation"])

print("\nUnivariate Analysis Report:")
print(report.to_string(index = False))

report.to_csv("univariate_analysis_report.csv", index = False)
print("\nReport saved successfully.")