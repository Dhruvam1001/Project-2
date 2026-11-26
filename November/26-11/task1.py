import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv("results_task2_25-11.csv")
required = ["time_on_app", "time_on_website"]

if not all(col in data.columns for col in required):
    raise ValueError("App time or website time column is missing.")

data[required] = data[required].apply(pd.to_numeric, errors = "coerce").fillna(0)
data["Total_Engagement"] = (data["time_on_app"] + data["time_on_website"])
data["App_Web_Ratio"] = (data["time_on_app"] / (data["time_on_website"] + 1))
data["Engagement_Index"] = (data["time_on_app"] * data["time_on_website"])
poly = PolynomialFeatures(degree = 2, include_bias = False)
features = poly.fit_transform(data[required])
poly_data = pd.DataFrame(features, columns = poly.get_feature_names_out(required), index = data.index)

for col in poly_data.columns:
    if col not in data.columns:
        data[col + "_poly"] = poly_data[col]

data.to_csv("results_task1.csv", index = False)

print("New engagement features created!")
print(data[["Total_Engagement", "App_Web_Ratio", "Engagement_Index"]].head())