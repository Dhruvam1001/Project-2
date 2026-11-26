import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
from scipy.stats import pearsonr, spearmanr

data = pd.read_csv("womens_clothing_segments.csv")
data["sentiment_score"] = data["description"].fillna("").apply(lambda text: TextBlob(str(text)).sentiment.polarity)
data["sentiment"] = data["sentiment_score"].apply(lambda x: "Positive" if x > 0 else "Negative" if x < 0 else "Neutral")
data = data.dropna(subset = ["ratings", "sentiment_score"])
pearson, p1 = pearsonr(data["ratings"], data["sentiment_score"])
spearman, p2 = spearmanr(data["ratings"], data["sentiment_score"])

print("Pearson Correlation:", round(pearson, 3))
print("Pearson p-value:", round(p1, 4))
print("Spearman Correlation:", round(spearman, 3))
print("Spearman p-value:", round(p2, 4))
print("\nSentiment Count:")
print(data["sentiment"].value_counts())

positive_feedback = data[data["sentiment"] == "Positive"].groupby("ratings").size()

print("\nPositive Feedback Count by Rating:")
print(positive_feedback)
print("\nAverage Sentiment by Rating:")
print(data.groupby("ratings")["sentiment_score"].mean().round(3))

if p1 < 0.05 and p2 < 0.05:
    print("\nSignificant correlation found.")
else:
    print("\nNo significant correlation found in both tests.")

data.to_csv("results_task2.csv", index = False)

print("\nResults saved successfully!")