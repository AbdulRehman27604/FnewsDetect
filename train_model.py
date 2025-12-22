import os
import nltk
import kagglehub
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB


stop_words = set(stopwords.words('english'))

path = kagglehub.dataset_download("saurabhshahane/fake-news-classification")

df = pd.read_csv(os.path.join(path, "WELFake_Dataset.csv"))
df.drop(columns=["Unnamed: 0", "title"], inplace=True)

df = df.sample(n=5000, random_state=42)

new_df = pd.DataFrame(columns=["text", "label"])

def clean_text(text):
    tokens = word_tokenize(text)
    tokens = [
        word.lower() for word in tokens
        if word.isalpha() and word.lower() not in stop_words
    ]
    return " ".join(tokens)

for row in df.itertuples(index=False):
    if pd.isna(row.text):
        continue

    cleaned_text = clean_text(row.text)
    label = row.label
    new_df.loc[len(new_df)] = [cleaned_text, label]

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(new_df["text"])
Y = new_df["label"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))


new_x = "Former President Barack Obama spoke at a climate conference in Paris on renewable energy initiatives."
cleaned_x = clean_text(new_x)
new_x_vec = vectorizer.transform([cleaned_x])
new_y = model.predict(new_x_vec)
if new_y == '1':
    print("True News")
else:
    print("False News")

trained_model = model
trained_vectorizer = vectorizer