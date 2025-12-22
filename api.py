from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from train_model import trained_model, trained_vectorizer, clean_text

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
