import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("results_task1.csv")
required = ["time_on_app", "time_on_website"]

if not all(col in data.columns for col in required):
    raise ValueError("App time or website time column is missing.")

data[required] = data[required].apply(pd.to_numeric, errors = "coerce")
data[required] = data[required].fillna(0)
data["Total_Engagement_Time"] = data[required].sum(axis=1)
scaler = MinMaxScaler()
X = scaler.fit_transform(data[["Total_Engagement_Time"]])

data["Engagement_Score"] = X.flatten() * 100
model = KMeans(n_clusters = 3, random_state = 42, n_init = 10)
data["Engagement_Cluster"] = model.fit_predict(X)
high_cluster = model.cluster_centers_.flatten().argmax()
data["Highly_Engaged"] = (data["Engagement_Cluster"] == high_cluster)
threshold = data["Total_Engagement_Time"].mean()

print("Average Engagement Time:", round(threshold, 2))
print(data[["Total_Engagement_Time", "Engagement_Score", "Highly_Engaged"]].head())

data.to_csv("results_task2.csv", index = False)