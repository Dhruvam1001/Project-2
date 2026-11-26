import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = pd.read_csv("womens_clothing_segments.csv")

X = data[["age"]].values
y = data["yearly amount spent"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

transformations = {
    "Linear": lambda x: x,
    "Polynomial 2": lambda x: np.c_[x, x**2],
    "Polynomial 3": lambda x: np.c_[x, x**2, x**3],
    "Log": lambda x: np.log(x),
    "Square Root": lambda x: np.sqrt(x)
}

results = {}
best_score = -float("inf")

for name, transform in transformations.items():
    model = LinearRegression()
    model.fit(transform(X_train), y_train)
    prediction = model.predict(transform(X_test))
    score = r2_score(y_test, prediction)
    results[name] = score
    print(name, "R2 Score:", round(score, 3))

    if score > best_score:
        best_score = score
        best_model = model
        best_name = name
        best_transform = transform

print("\nBest Model:", best_name)
print("Best R2 Score:", round(best_score, 3))

data["Age_Group"] = pd.cut(data["age"], bins = [17, 30, 45, 65], labels = ["Young Adults", "Middle Aged", "Older Adults"])

plt.figure(figsize=(9, 5))

for group, part in data.groupby("Age_Group", observed = True):
    plt.scatter(part["age"], part["yearly amount spent"], alpha = 0.4, label = group)

ages = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)

plt.plot(ages, best_model.predict(best_transform(ages)), color = "black", linewidth = 2, label = "Best Regression")
plt.xlabel("Customer Age")
plt.ylabel("Yearly Amount Spent")
plt.title("Age vs Yearly Spending")
plt.legend()
plt.grid()
plt.show()

data.to_csv("results_task1.csv", index = False)