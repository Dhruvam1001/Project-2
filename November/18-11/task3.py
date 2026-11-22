import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

data = pd.read_csv("womens_clothing_standardized.csv")
cleaned = data.copy()
cols = [c for c in data.select_dtypes(include = "number").columns if not c.endswith("id") and c != "target"]
scaled = StandardScaler().fit_transform(data[cols].fillna(data[cols].median()))
dbscan_out = DBSCAN(eps = 1.5, min_samples = 5).fit_predict(scaled) == -1
log = []
remove_rows = pd.Series(False, index = data.index)
treatments = {"price_inr": "transform", "numimages": "cap"}

for col in cols:
    q1 = data[col].quantile(0.25)
    q3 = data[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    iqr_out = (data[col] < lower) | (data[col] > upper)
    z = (data[col] - data[col].mean()) / (data[col].std() + 1e-8)
    z_out = z.abs() > 3
    outliers = iqr_out | z_out | dbscan_out
    action = treatments.get(col, "cap")

    if action == "cap":
        cleaned[col] = cleaned[col].clip(lower, upper)

    elif action == "transform":
        cleaned[col] = np.sign(cleaned[col]) * np.log1p(np.abs(cleaned[col]))

    elif action == "remove":
        remove_rows |= outliers

    log.append([col, int(iqr_out.sum()), int(z_out.sum()), int((dbscan_out & outliers).sum()), action])

cleaned = cleaned.loc[~remove_rows]

print("\nOutlier Detection and Treatment Report")
print(pd.DataFrame(log, columns = [ "Feature", "IQR Outliers", "Z-Score Outliers", "DBSCAN Anomalies", "Treatment"]).to_string(index = False))
print("\nTotal DBSCAN Anomalies:", int(dbscan_out.sum()))
print("Original Rows:", len(data))
print("Removed Rows:", int(remove_rows.sum()))
print("Final Rows:", len(cleaned))

cleaned.to_csv("womens_clothing_outliers_treated.csv", index = False)

print("\nCleaned dataset saved successfully.")