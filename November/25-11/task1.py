import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download("vader_lexicon")

data = pd.read_csv("results_task3.csv")
sia = SentimentIntensityAnalyzer()
data["Sentiment_Score"] = data["description"].fillna("").apply(lambda text: sia.polarity_scores(str(text))["compound"])
data["Normalized_Sentiment"] = (data["Sentiment_Score"] + 1) / 2

print(data[["description", "Sentiment_Score", "Normalized_Sentiment"]].head())

data.to_csv("results_task1.csv", index = False)