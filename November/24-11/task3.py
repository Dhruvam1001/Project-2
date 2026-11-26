import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

data = pd.read_csv("results_task2.csv")
data["age"] = pd.to_numeric(data["age"], errors = "coerce")
data = data.dropna(subset = ["age"]).copy()

X = data[["age"]]

best_score = -1

for k in range(2, min(7, data["age"].nunique())):
    model = KMeans(n_clusters = k, random_state = 42, n_init = 10)
    labels = model.fit_predict(X)
    score = silhouette_score(X, labels)

    if score > best_score:
        best_score = score
        best_model = model

data["Age_Group"] = best_model.predict(X)
centers = best_model.cluster_centers_.flatten()
order = centers.argsort()
group_names = {old: f"Age Group {new + 1}" for new, old in enumerate(order)}
data["Age_Group"] = data["Age_Group"].map(group_names)

print("Optimal number of age groups:", len(centers))
print("Silhouette Score:", round(best_score, 3))
print("\nAge Group Counts:")
print(data["Age_Group"].value_counts())

data.to_csv("results_task3.csv", index = False)