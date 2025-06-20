# Fake-News-Detection

📰 Project Title: Fake News Detection using Scikit-learn (sklearn)
📋 Project Overview:
The Fake News Detection system is a machine learning project developed using Python's scikit-learn library. It aims to classify news articles as “Real” or “Fake” based on their textual content. The system leverages Natural Language Processing (NLP) techniques for feature extraction and applies supervised learning models to detect misinformation.

🎯 Objective:
To build an intelligent system that can automatically identify and flag fake news articles by training a machine learning model on labeled datasets of real and fake news.

🛠️ Tech Stack:
Component	Tool/Library
Programming Lang	Python
ML Library	Scikit-learn (sklearn)
NLP	CountVectorizer, Tfidf
Dataset Handling	Pandas, NumPy
Text Cleaning	re, nltk (optional)
Model Types	Logistic Regression, Naive Bayes, SVM, etc.

📦 Dataset:
Typically based on a CSV file containing:

text: News content

label: Binary value (1 = Real, 0 = Fake)

Public datasets include:

Kaggle Fake and Real News Dataset

LIAR dataset

🔍 Workflow:
1. Data Preprocessing:
Load dataset using pandas

Remove null values, punctuation, stopwords

Convert text to lowercase

Tokenization (optional)

2. Feature Extraction:
Convert textual data to numerical form using:

CountVectorizer() – Bag of Words model

or TfidfVectorizer() – TF-IDF representation

3. Model Building:
Split data using train_test_split()

Train models using scikit-learn classifiers such as:

LogisticRegression()

MultinomialNB()

PassiveAggressiveClassifier()

Evaluate using accuracy, confusion matrix, precision, recall, F1-score.

4. Prediction:
Accept new input news text

Preprocess and vectorize

Predict and output “Real” or “Fake”

📈 Evaluation Metrics:
Accuracy Score

Confusion Matrix

Precision, Recall, F1-score

ROC-AUC Curve (optional)

💡 Key Features:
Identifies misleading news articles using ML.

Uses only text-based features (no images or metadata).

Lightweight and easy to train on local machines.

Can be extended with deep learning or web-based UI.
