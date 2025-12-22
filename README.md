#  Fake News Detection AI

This project is an **AI-powered fake news detection system** that analyzes the content of news articles and predicts whether they are **REAL** or **FAKE** using natural language processing and machine learning techniques.

The goal of the project is to explore how text data can be processed and classified, and to build a complete end-to-end AI application with a web interface.

---

##  Features

-  Machine learning–based text classification
-  Cleans and preprocesses raw news text
-  Uses TF-IDF for feature extraction
-  Classifies news as **REAL** or **FAKE**
-  Web interface for user input and predictions
-  FastAPI backend for serving the model

---

##  Technologies Used

### Backend & AI
- Python
- Natural Language Processing (NLP)
- Scikit-learn
- TF-IDF Vectorization
- Multinomial Naive Bayes

### Web Framework
- FastAPI

### Frontend
- HTML
- CSS (basic styling)

---

##  Model Performance

- **Accuracy achieved:** **83%**
- Dataset used: Fake news classification dataset
- Evaluation method: Train-test split and classification metrics

>  Note:  
> This model analyzes **writing patterns and linguistic features**, not factual truth.  
> Predictions are probabilistic and may be incorrect for short, ambiguous, or formally written fake news.

---

##  How It Works

1. News text is cleaned and tokenized
2. Stopwords and non-alphabetic tokens are removed
3. Text is converted into numerical features using **TF-IDF**
4. A machine learning classifier predicts whether the news is real or fake
5. The result is displayed through a web interface

---


