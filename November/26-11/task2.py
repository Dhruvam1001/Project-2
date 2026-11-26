import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.feature_selection import SelectKBest, chi2
from nltk.stem import PorterStemmer

data = pd.read_csv("results_task1.csv")
data["description"] = data["description"].fillna("").astype(str)
data["ratings"] = pd.to_numeric(data["ratings"], errors = "coerce")
data = data.dropna(subset = ["ratings"]).copy()
stemmer = PorterStemmer()

def preprocess(text):
    words = text.lower().split()
    return " ".join(stemmer.stem(word) for word in words
        if word.isalpha() and word not in ENGLISH_STOP_WORDS)

data["clean_text"] = data["description"].apply(preprocess)
y = (data["ratings"] >= data["ratings"].median()).astype(int)
vectorizer = TfidfVectorizer(ngram_range = (1, 2), max_features = 1000)
X = vectorizer.fit_transform(data["clean_text"])
k = min(300, X.shape[1])
selector = SelectKBest(chi2, k=k)
X_selected = selector.fit_transform(X, y)
result = pd.DataFrame(X_selected.toarray(), columns = vectorizer.get_feature_names_out()[selector.get_support()])
result["ratings"] = data["ratings"].values
result.to_csv("results_task2.csv", index = False)

print("TF-IDF vectorization completed!")
print("Selected features:", X_selected.shape[1])
print(result.head())