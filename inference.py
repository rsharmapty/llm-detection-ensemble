import os
import joblib
import numpy as np
import pandas as pd
import torch
import nltk
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer
from datasets import Dataset
from feature_engineering import extract_statistical_features # Re-use feature extraction logic

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# --- Configuration ---
MODEL_DIR = 'models'
TFIDF_VECTORIZER_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')
LR_MODEL_PATH = os.path.join(MODEL_DIR, 'logistic_regression.pkl')
XGB_MODEL_PATH = os.path.join(MODEL_DIR, 'xgboost.pkl')
ROBERTA_MODEL_PATH = os.path.join(MODEL_DIR, 'roberta_model')

# --- Load Models and Vectorizer ---
print("Loading all models and vectorizers for inference...")
tfidf_vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
lr_model = joblib.load(LR_MODEL_PATH)
xgb_model = joblib.load(XGB_MODEL_PATH)
roberta_model = RobertaForSequenceClassification.from_pretrained(ROBERTA_MODEL_PATH)
roberta_tokenizer = RobertaTokenizer.from_pretrained(ROBERTA_MODEL_PATH)
print("All models loaded successfully.")

def predict(text_input):
    """
    Analyzes a new text input and detects its origin using all four strategies.

    Args:
        text_input (str): The text to be analyzed.

    Returns:
        dict: A dictionary containing the prediction from each model.
    """
    print(f"\n--- Analyzing new text ---")
    # Create a DataFrame for consistent processing
    df = pd.DataFrame([{'text': text_input}])
    
    # 1. Logistic Regression Prediction
    text_tfidf = tfidf_vectorizer.transform(df['text'])
    lr_pred = lr_model.predict(text_tfidf)[0]
    
    # 2. XGBoost Prediction
    text_stats = extract_statistical_features(df)
    xgb_pred = xgb_model.predict(text_stats)[0]
    
    # 3. RoBERTa Prediction
    dataset = Dataset.from_pandas(df)
    tokenized_dataset = dataset.map(lambda e: roberta_tokenizer(e['text'], padding='max_length', truncation=True, max_length=256), batched=True)
    trainer = Trainer(model=roberta_model)
    predictions = trainer.predict(tokenized_dataset)
    roberta_pred = np.argmax(predictions.predictions, axis=1)[0]
    
    # 4. Ensemble Prediction (Majority Vote)
    votes = [lr_pred, xgb_pred, roberta_pred]
    ensemble_pred = max(set(votes), key=votes.count)
    
    # Map predictions to human-readable labels
    label_map = {0: "Human-Written", 1: "AI-Generated"}
    
    results = {
        "Logistic Regression": label_map[lr_pred],
        "XGBoost (Statistical)": label_map[xgb_pred],
        "RoBERTa": label_map[roberta_pred],
        "Ensemble (Final Verdict)": label_map[ensemble_pred]
    }
    
    return results

if __name__ == '__main__':
    # --- Example Usage ---
    
    # Example 1: A typical human-like text
    human_text = "I went to the store yesterday to buy some groceries. The weather was really nice, so I decided to walk instead of taking the bus. It felt good to get some fresh air."
    
    # Example 2: A more formal, AI-like text
    ai_text = "The utilization of advanced computational models facilitates the optimization of logistical networks, thereby enhancing efficiency and reducing operational overhead across various industrial sectors."
    
    # Run predictions
    human_results = predict(human_text)
    print("\n--- Prediction Results for Human Text ---")
    for model_name, prediction in human_results.items():
        print(f"{model_name}: {prediction}")
        
    ai_results = predict(ai_text)
    print("\n--- Prediction Results for AI Text ---")
    for model_name, prediction in ai_results.items():
        print(f"{model_name}: {prediction}")

    # Example 3: Test with your own text
    your_text = input("\n--- Enter your own text to analyze: --- \n")
    if your_text:
        your_results = predict(your_text)
        print("\n--- Prediction Results for Your Text ---")
        for model_name, prediction in your_results.items():
            print(f"{model_name}: {prediction}")
