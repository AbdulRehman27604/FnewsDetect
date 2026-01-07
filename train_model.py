import os
import nltk
import kagglehub
import pandas as pd
import joblib

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Download NLTK resources
nltk.download("punkt")
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

def clean_text(text):
    tokens = word_tokenize(text)
    tokens = [
        word.lower()
        for word in tokens
        if word.isalpha() and word.lower() not in stop_words
    ]
    return " ".join(tokens)

# Download dataset
path = kagglehub.dataset_download("saurabhshahane/fake-news-classification")
df = pd.read_csv(os.path.join(path, "WELFake_Dataset.csv"))

# Clean dataset
df.drop(columns=["Unnamed: 0", "title"], inplace=True)
df = df.sample(n=5000, random_state=42)
df = df.dropna(subset=["text"])

# Preprocess text
df["text"] = df["text"].apply(clean_text)

X = df["text"]
y = df["label"]

# Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save artifacts
joblib.dump(model, "model.joblib")
joblib.dump(vectorizer, "vectorizer.joblib")

print("Training complete. Model and vectorizer saved.")
