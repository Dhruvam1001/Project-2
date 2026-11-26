import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from statsmodels.stats.outliers_influence import variance_inflation_factor

data = pd.read_csv("womens_clothing_segments.csv")

X = data[["time_on_app", "time_on_website"]]
y = data["yearly amount spent"]

data = data.dropna(subset = ["time_on_app", "time_on_website", "yearly amount spent"])
X = data[["time_on_app", "time_on_website"]]
y = data["yearly amount spent"]

vif = pd.DataFrame()
vif["Feature"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print("Multicollinearity (VIF):\n", vif.round(3))

scaler = StandardScaler()
scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters = 3, random_state = 42, n_init = 10)
data["Behavior_Cluster"] = kmeans.fit_predict(scaled)

print("\nCustomer Behavior Clusters:")
print(data.groupby("Behavior_Cluster")[["time_on_app", "time_on_website", "yearly amount spent"]].mean().round(2))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

model = RandomForestRegressor(n_estimators = 100, random_state = 42)
model.fit(X_train, y_train)
prediction = model.predict(X_test)

print("\nR2 Score:", round(r2_score(y_test, prediction), 3))
print("MAE:", round(mean_absolute_error(y_test, prediction), 2))

importance = pd.Series(model.feature_importances_, index = X.columns).sort_values(ascending = False)

print("\nFeature Importance:")
print(importance.round(3))
print("\nStronger Predictive Factor:", importance.idxmax())

data.to_csv("results_task3.csv", index = False)

print("\nResults saved successfully!")