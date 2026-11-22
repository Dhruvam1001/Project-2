import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("womens_clothing_scaled.csv")
cols = data.select_dtypes(include = "object").columns.tolist()
cols = [c for c in cols if "id" not in c.lower() and data[c].nunique() / len(data) <= 0.8]

frequency = []
entropy = []
encoded = data.copy()

for col in cols:
    counts = data[col].fillna("Missing").value_counts()
    probabilities = counts / counts.sum()

    for category, count in counts.items():
        frequency.append([col, category, count, round(count / len(data) * 100, 2)])

    score = -(probabilities * np.log2(probabilities)).sum()
    normalized = score / np.log2(len(counts)) if len(counts) > 1 else 0
    entropy.append([col, round(score, 3), round(normalized, 3)])

    if data[col].nunique() > 20:
        encoded[col + "_encoded"] = data[col].map(data[col].value_counts(normalize = True)).fillna(0)

freq_report = pd.DataFrame(frequency, columns = ["Feature", "Category", "Count", "Percentage"])

print("\nFrequency Distribution:")
print(freq_report.to_string(index = False))

entropy_report = pd.DataFrame(entropy, columns = ["Feature", "Entropy", "Normalized Entropy"]).sort_values("Normalized Entropy", ascending = False)

print("\nEntropy-Based Category Importance:")
print(entropy_report.to_string(index = False))

if "PrimaryColor" in cols:
    counts = data["PrimaryColor"].fillna("Missing").value_counts()
    counts.head(5).plot(kind = "pie", autopct = "%1.1f%%")
    plt.title("Top 5 Primary Colors")
    plt.ylabel("")
    plt.show()

if "ProductBrand" in cols and "PrimaryColor" in cols:
    top_brands = data["ProductBrand"].value_counts().head(5).index
    top_colors = data["PrimaryColor"].value_counts().head(4).index
    sample = data[data["ProductBrand"].isin(top_brands) & data["PrimaryColor"].isin(top_colors)]
    table = pd.crosstab(sample["ProductBrand"], sample["PrimaryColor"])
    table.plot(kind = "bar", stacked = True, figsize = (9, 5))

    plt.title("Product Brand vs Primary Color")
    plt.xlabel("Product Brand")
    plt.ylabel("Number of Products")
    plt.xticks(rotation = 45)
    plt.tight_layout()
    plt.show()

freq_report.to_csv("categorical_frequency.csv", index = False)
entropy_report.to_csv("category_entropy.csv", index = False)
encoded.to_csv("womens_clothing_categorical_encoded.csv", index = False)

print("\nCategorical analysis completed successfully.")