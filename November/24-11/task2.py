import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
from textstat import flesch_reading_ease
from scipy.stats import pearsonr

data = pd.read_csv("womens_clothing_segments.csv")
data["description"] = data["description"].fillna("").astype(str)
data["word_count"] = data["description"].apply(lambda x: len(x.split()))
data["character_count"] = data["description"].apply(len)
data["avg_word_length"] = data["description"].apply(lambda x: sum(len(w) for w in x.split()) / max(len(x.split()), 1))
data["sentiment"] = data["description"].apply(lambda x: TextBlob(x).sentiment.polarity)
data["sentiment_shift"] = data["description"].apply(lambda x: pd.Series([s.sentiment.polarity for s in TextBlob(x).sentences]).std())
data["sentiment_shift"] = data["sentiment_shift"].fillna(0)
data["readability"] = data["description"].apply(lambda x: flesch_reading_ease(x) if x.strip() else 0)
features = ["word_count", "character_count", "avg_word_length", "sentiment", "sentiment_shift", "readability"]

for col in features:
    correlation, p_value = pearsonr(data[col], data["ratings"])
    print(col, "Correlation:", round(correlation, 3), "P-value:", round(p_value, 4))

data.to_csv("results_task2.csv", index = False)