import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, kendalltau

data = pd.read_csv("womens_clothing_segments.csv")

X = data.select_dtypes(include = "number")
X = X.drop(columns = [c for c in X.columns if "id" in c.lower() or c.lower() == "segment"], errors = "ignore")
X = X.loc[:, X.nunique() > 1].fillna(X.median())

if X.shape[1] < 2:
    raise ValueError("At least two numerical features are required.")

methods = {
    "Pearson": pearsonr,
    "Spearman": spearmanr,
    "Kendall": kendalltau
}

all_results = []

for name, method in methods.items():
    corr = pd.DataFrame(np.eye(len(X.columns)), index = X.columns, columns = X.columns)
    pvalues = corr.copy()

    for i, c1 in enumerate(X.columns):
        for j, c2 in enumerate(X.columns):
            if i < j:
                r, p = method(X[c1], X[c2])
                corr.loc[c1, c2] = corr.loc[c2, c1] = r
                pvalues.loc[c1, c2] = pvalues.loc[c2, c1] = p

                if p < 0.05:
                    all_results.append([name, c1, c2, round(r, 3), round(p, 5)])

    sns.clustermap(corr, cmap = "coolwarm", center = 0, fmt = ".2f", figsize = (8, 6))
    plt.suptitle(name + " Correlation Heatmap")
    plt.savefig(name.lower() + "_heatmap.png")
    plt.show()
    plt.close()

report = pd.DataFrame(all_results, columns = ["Method", "Feature 1", "Feature 2", "Correlation", "P-Value"])

print("\nSignificant Correlations (p < 0.05):")
print(report.to_string(index = False) if not report.empty else "No significant correlations found.")

report.to_csv("significant_correlations.csv", index = False)

print("\nCorrelation analysis completed.")