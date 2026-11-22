import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

data = pd.read_csv("womens_clothing_outliers_treated.csv")
cols = [c for c in data.select_dtypes(include = "number").columns if "id" not in c.lower() and c.lower() != "target"]

X = data[cols].fillna(data[cols].median())

scalers = {
    "MinMax": MinMaxScaler(),
    "Standard": StandardScaler(),
    "Robust": RobustScaler(),
    "Power": PowerTransformer(method = "yeo-johnson")
}

results = {}
report = []

for name, scaler in scalers.items():
    scaled = scaler.fit_transform(X)
    clusters = KMeans(n_clusters = 2, random_state = 42, n_init = 10).fit_predict(scaled)
    score = silhouette_score(scaled, clusters)
    skewness = pd.DataFrame(scaled, columns = cols).skew().abs().mean()
    results[name] = scaled
    report.append([name, round(score, 3), round(skewness, 3)])

report = pd.DataFrame(report, columns = ["Scaling Method", "Silhouette Score", "Average Absolute Skewness"])
report["Performance Rank"] = report["Silhouette Score"].rank(ascending = False)
report["Distribution Rank"] = report["Average Absolute Skewness"].rank()
report["Average Rank"] = report[["Performance Rank", "Distribution Rank"]].mean(axis=1)
best = report.loc[report["Average Rank"].idxmin(), "Scaling Method"]
data[cols] = results[best]
data.to_csv("womens_clothing_scaled.csv", index = False)

print("\nScaling Comparison:")
print(report.to_string(index = False))
print("\nSelected Scaling Method:", best)
print("\nScaled Dataset Saved: womens_clothing_scaled.csv")