import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

data = pd.read_csv("womens_clothing_categorical_encoded.csv")

X = data[["price_inr", "numimages"]].dropna()
scaled = StandardScaler().fit_transform(X)
models = [GaussianMixture(n_components = k, random_state = 42).fit(scaled) for k in [2, 3, 4]]
model = min(models, key = lambda m: m.bic(scaled))
data["Segment"] = model.predict(StandardScaler().fit(X).transform(data[["price_inr", "numimages"]]))
scores = model.score_samples(scaled)
data["Anomaly"] = False
data.loc[X.index, "Anomaly"] = (scores < pd.Series(scores).quantile(0.01).item())
report = data.groupby("Segment").agg(Products = ("productid", "count"), AveragePrice = ("price_inr", "mean"), AverageImages = ("numimages", "mean")).round(3)

print("\nProduct Segmentation:")
print(report)
print("\nTop Brands:")
print(data["productbrand"].value_counts().head(10))
print("\nPopular Colours:")
print(data["primarycolor"].value_counts().head(10))

anomalies = data[data["Anomaly"]]

print("\nDetected Anomalies:", len(anomalies))
print(anomalies[["productname", "productbrand", "price_inr"]].head(10))

plt.scatter(X["price_inr"], X["numimages"], c = model.predict(scaled), cmap = "viridis")
plt.xlabel("Product Price")
plt.ylabel("Number of Images")
plt.title("GMM Product Segmentation")
plt.show()

data["price_inr"].hist(bins = 20)

plt.title("Product Price Distribution")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

report.to_csv("product_segment_report.csv")
data.to_csv("womens_clothing_segments.csv", index = False)
anomalies.to_csv("product_anomalies.csv", index = False)

print("\nAnalysis Completed Successfully.")