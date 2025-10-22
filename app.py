import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer
from datasets import Dataset
import torch

# Local imports from your existing project structure
from feature_engineering import create_statistical_features_for_single_text, gpt2_tokenizer as perplexity_tokenizer

# --- 1. Initialize Flask App ---
app = Flask(__name__)

# --- 2. Load All Models and necessary components at startup ---
MODEL_DIR = 'models'
print("--- Loading models and components... ---")

# Load classic ML models
try:
    lr_model = joblib.load(os.path.join(MODEL_DIR, 'logistic_regression.pkl'))
    xgb_model = joblib.load(os.path.join(MODEL_DIR, 'xgboost.pkl'))
    tfidf_vectorizer = joblib.load(os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))
    print("Classic ML models and TF-IDF vectorizer loaded successfully.")
except FileNotFoundError:
    print("Error: Classic ML models or TF-IDF vectorizer not found. Please run main.py to train them first.")
    lr_model = xgb_model = tfidf_vectorizer = None

# Load RoBERTa model
try:
    roberta_model_path = os.path.join(MODEL_DIR, 'roberta_model')
    roberta_model = RobertaForSequenceClassification.from_pretrained(roberta_model_path)
    roberta_tokenizer = RobertaTokenizer.from_pretrained(roberta_model_path)
    print("RoBERTa model loaded successfully.")
except Exception as e:
    print(f"Error loading RoBERTa model: {e}. Please ensure it's trained and saved correctly.")
    roberta_model = roberta_tokenizer = None

print("--- Model loading complete. API is ready. ---")

# --- 3. Define API Endpoints ---

@app.route('/')
def index():
    """Renders the main web page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Receives text and returns predictions from all models."""
    if not all([lr_model, xgb_model, tfidf_vectorizer, roberta_model, roberta_tokenizer, perplexity_tokenizer]):
        return jsonify({"error": "Models are not loaded. Please train the models by running main.py."}), 500

    data = request.get_json()
    if 'text' not in data or not data['text'].strip():
        return jsonify({"error": "No text provided."}), 400

    text_to_analyze = data['text']

    # --- Prediction Logic ---
    results = {}

    # 1. Logistic Regression Prediction
    text_tfidf = tfidf_vectorizer.transform([text_to_analyze])
    lr_prob = lr_model.predict_proba(text_tfidf)[0]
    lr_pred_label = "AI" if lr_prob[1] > 0.5 else "Human"
    lr_confidence = lr_prob[1] if lr_pred_label == "AI" else lr_prob[0]
    results['lr'] = {'prediction': lr_pred_label, 'confidence': f"{lr_confidence:.2%}"}

    # 2. XGBoost Prediction
    statistical_features = create_statistical_features_for_single_text(text_to_analyze)
    xgb_prob = xgb_model.predict_proba(statistical_features)[0]
    xgb_pred_label = "AI" if xgb_prob[1] > 0.5 else "Human"
    xgb_confidence = xgb_prob[1] if xgb_pred_label == "AI" else xgb_prob[0]
    results['xgb'] = {'prediction': xgb_pred_label, 'confidence': f"{xgb_confidence:.2%}"}

    # 3. RoBERTa Prediction
    inputs = roberta_tokenizer(text_to_analyze, return_tensors="pt", padding=True, truncation=True, max_length=256)
    with torch.no_grad():
        logits = roberta_model(**inputs).logits
    
    probabilities = torch.softmax(logits, dim=1).squeeze()
    roberta_prob_ai = probabilities[1].item()
    roberta_pred_label = "AI" if roberta_prob_ai > 0.5 else "Human"
    roberta_confidence = roberta_prob_ai if roberta_pred_label == "AI" else 1 - roberta_prob_ai
    results['roberta'] = {'prediction': roberta_pred_label, 'confidence': f"{roberta_confidence:.2%}"}

    return jsonify(results)

if __name__ == '__main__':
    # Creates the 'templates' folder if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # Check if index.html exists, if not, create a placeholder
    if not os.path.exists('templates/index.html'):
        with open('templates/index.html', 'w') as f:
            f.write("<html><body><h1>Placeholder</h1><p>This is a placeholder. The actual index.html file with the UI should be placed here.</p></body></html>")
            print("Created a placeholder index.html. Please replace it with the full frontend code.")

    app.run(debug=True, port=5001)
