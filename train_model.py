import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

fake = pd.read_csv("data/Fake.csv")
real  = pd.read_csv("data/True.csv")

fake["label"] = 0
real["label"] = 1

data = pd.concat([fake,real], ignore_index=True)

data["title"] = data["title"].fillna("")
data["text"] = data["text"].fillna("")
data["content"] = data["title"] + " " + data["text"]

print(data["label"].value_counts())
print(data.groupby("label")["content"].count())

data = data[data["content"].str.strip() != ""]

data = data.drop_duplicates(subset=["content"])

x = data["content"]
y = data["label"]

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=100000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )),
    ("classifier", LogisticRegression(
        C=1.0,
        max_iter=2000,
        class_weight="balanced"
    ))
])



model.fit(x_train, y_train)

predictions = model.predict(x_test)

print("\nAccuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report: ")
print(classification_report(
    y_test,
    predictions,
    target_names=["Fake","Real"]
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

joblib.dump(model, "fake_news_model.joblib")

print("\nModel Saved Successfully")