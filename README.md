# Fake News Detection System

## Project Overview

The Fake News Detection System is a Machine Learning and Natural Language Processing (NLP) based application that classifies news articles as **Real News** or **Fake News**. The system analyzes the textual content of news articles and predicts their authenticity using a trained machine learning model.

This project helps users quickly identify potentially misleading or fake news articles based on patterns learned from historical news datasets.

---

## Objectives

* Detect fake and real news articles automatically.
* Apply Natural Language Processing techniques for text analysis.
* Build a machine learning model with high prediction accuracy.
* Develop an interactive web application using Streamlit.

---

## Features

*  Fake News Detection
*  Real News Classification
*  TF-IDF Text Vectorization
*  Logistic Regression Model
*  Confidence Score Display
*  User-Friendly Streamlit Interface
*  Real-Time Prediction
*  Word Count Analysis

---

## Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* Scikit-learn
* Streamlit
* Pickle
* Regular Expressions (re)

### Machine Learning

* Logistic Regression

### Natural Language Processing

* TF-IDF Vectorization

---

## Dataset

The project uses two datasets:

### Fake.csv

Contains fake news articles.

### True.csv

Contains genuine news articles.

Dataset fields include:

* Title
* Text
* Subject
* Date

---

## Methodology

### 1. Data Collection

* Load Fake.csv and True.csv datasets.

### 2. Data Preprocessing

* Convert text to lowercase.
* Remove special characters.
* Remove unwanted spaces.

### 3. Feature Extraction

* Convert text into numerical vectors using TF-IDF.

### 4. Model Training

* Train Logistic Regression classifier.

### 5. Evaluation

* Test model performance on unseen data.

### 6. Deployment

* Build a Streamlit web application for user interaction.

---

## Model Performance

**Algorithm:** Logistic Regression

**Feature Extraction:** TF-IDF Vectorizer

**Accuracy Achieved:** 98.7%

---

## Project Structure

Fake News Detector/

├── Fake.csv

├── True.csv

├── train_model.py

├── app.py

├── model.pkl

├── vectorizer.pkl

├── README.md

└── screenshots/

---

## ▶️ How to Run

### Step 1: Install Required Packages

```bash
pip install pandas scikit-learn streamlit
```

### Step 2: Train the Model

```bash
python train_model.py
```

This generates:

* model.pkl
* vectorizer.pkl

### Step 3: Launch the Application

```bash
streamlit run app.py
```

---

## How It Works

1. User enters a news article.
2. Text is cleaned and preprocessed.
3. TF-IDF converts text into numerical vectors.
4. Logistic Regression predicts whether the article is Real or Fake.
5. Prediction and confidence score are displayed.

---

## Sample Output

### Real News

✅ REAL NEWS
![Real News](screenshots/real news.png)

### Fake News

❌ FAKE NEWS
![Fake News](screenshots/fake news.png)

---

## Future Enhancements

* Deep Learning Models (LSTM, BERT)
* Multi-Language News Detection
* Real-Time News API Integration
* Explainable AI (XAI)
* News Source Credibility Analysis
* Fake News Visualization Dashboard

---

## Developed By

Dharshini Mary J

Artificial Intelligence and Data Science (AIDS)

Machine Learning & NLP Project

---

## License

This project is developed for educational and academic purposes.
# Fake-news-prediction-system
