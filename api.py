import nltk
import joblib

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Safe NLTK setup (Render-compatible)
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

stop_words = set(stopwords.words("english"))

def clean_text(text):
    tokens = word_tokenize(text)
    tokens = [
        word.lower()
        for word in tokens
        if word.isalpha() and word.lower() not in stop_words
    ]
    return " ".join(tokens)

# Load trained model and vectorizer
trained_model = joblib.load("model.joblib")
trained_vectorizer = joblib.load("vectorizer.joblib")

app = FastAPI(title="Fake News Detection App")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": None}
    )

@app.post("/predict", response_class=HTMLResponse)
def predict(request: Request, news: str = Form(...)):
    cleaned = clean_text(news)
    vector = trained_vectorizer.transform([cleaned])
    prediction = trained_model.predict(vector)[0]

    result = "REAL NEWS ✅" if prediction == 1 else "FAKE NEWS ❌"

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": result}
    )
