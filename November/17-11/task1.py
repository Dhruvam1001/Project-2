import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

data = pd.read_csv("womens_clothing.csv")
data["PrimaryColor"] = data["PrimaryColor"].str.strip()
missing = data["PrimaryColor"].isna()
colors = sorted(data["PrimaryColor"].dropna().unique())

X = data[["ProductBrand", "Gender", "Price (INR)", "NumImages"]].copy()
X["BrandFrequency"] = X["ProductBrand"].map(X["ProductBrand"].value_counts(normalize = True))
X = pd.concat([X.drop(columns = "ProductBrand"), pd.get_dummies(X["Gender"], dtype = int)], axis=1)
X = X.drop(columns = "Gender")
X = StandardScaler().fit_transform(X)
Y = pd.get_dummies(data["PrimaryColor"]).reindex(columns = colors, fill_value = 0).astype(float)
Y.loc[missing] = np.nan
n = X.shape[1]

combined = np.column_stack([X, Y])
known = np.where(~missing)[0]
train, test = train_test_split(known, test_size = 0.2, random_state = 42)
actual = data.loc[test, "PrimaryColor"].values
validation = combined.copy()
validation[test, n:] = np.nan
knn = KNNImputer(n_neighbors = 5)
knn_result = knn.fit_transform(validation)
knn_pred = np.array(colors)[np.argmax(knn_result[test, n:], axis=1)]
mice_results = []

for seed in [1, 2, 3]:
    mice = IterativeImputer(estimator = BayesianRidge(), max_iter = 5, sample_posterior = True, random_state = seed)
    result = mice.fit_transform(validation)
    mice_results.append(result)

mice_pred = np.array(colors)[np.argmax(np.mean(mice_results, axis=0)[test, n:], axis=1)]
model = RandomForestClassifier(n_estimators = 50, random_state = 42, n_jobs = -1)
model.fit(X[train], data.loc[train, "PrimaryColor"])
predictive_pred = model.predict(X[test])
predictions = {
    "KNN": knn_pred,
    "MICE": mice_pred,
    "Predictive Modeling": predictive_pred
}

results = []

for method, pred in predictions.items():
    results.append([method, round(accuracy_score(actual, pred), 3), round(f1_score(actual, pred, average = "macro", zero_division = 0), 3)])

report = pd.DataFrame(results, columns = ["Method", "Accuracy", "Macro F1"])

print("\nImputation Comparison")
print(report.to_string(index=False))

best = report.loc[report["Macro F1"].idxmax(), "Method"]
print("\nBest Imputation Method:", best)

if best == "Predictive Modeling":
    model.fit(X[~missing], data.loc[~missing, "PrimaryColor"])
    data.loc[missing, "PrimaryColor"] = model.predict(X[missing.to_numpy()])

else:
    if best == "KNN":
        final = KNNImputer(n_neighbors = 5).fit_transform(combined)
    else:
        final = np.mean([IterativeImputer(estimator = BayesianRidge(), max_iter = 5, sample_posterior = True, random_state = s).fit_transform(combined)
            for s in [1, 2, 3]], axis=0)

    data.loc[missing, "PrimaryColor"] = np.array(colors)[np.argmax(final[missing.to_numpy(), n:], axis=1)]

data.to_csv("womens_clothing_imputed.csv", index = False)

print("\nMissing Values Before:", missing.sum())
print("Missing Values After:", data["PrimaryColor"].isna().sum())
print("Saved: womens_clothing_imputed.csv")