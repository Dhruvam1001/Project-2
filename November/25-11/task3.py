import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("results_task2.csv")
target = "yearly amount spent"
cat_cols = data.select_dtypes(include = ["str", "object", "category"]).columns

for col in cat_cols:
    if col == target:
        continue

    if data[col].nunique() <= 10:
        data = pd.get_dummies(data, columns = [col], prefix = col, dtype = int)
    else:
        freq = data[col].value_counts(normalize = True)
        data[col] = data[col].map(lambda x: x if freq.get(x, 0) >= 0.01 else "Other")
        data[col + "_target"] = data[col].map(data.groupby(col)[target].mean())
        data[col + "_frequency"] = data[col].map(data[col].value_counts(normalize = True))
        data.drop(columns = [col], inplace = True)

num_cols = data.select_dtypes(include = "number").columns.drop(target, errors = "ignore")

if len(num_cols) > 1:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data[num_cols].fillna(0))
    pca = PCA(n_components = 0.95)
    reduced = pca.fit_transform(scaled)
    pca_data = pd.DataFrame(reduced, columns = [f"PCA_{i+1}" for i in range(reduced.shape[1])])
    data = pd.concat([data.drop(columns = num_cols).reset_index(drop = True), pca_data, data[[target]].reset_index(drop = True)], axis=1)

data.to_csv("results_task3.csv", index = False)

print("Encoding completed successfully!")
print("Final dataset shape:", data.shape)