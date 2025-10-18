import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

MODEL_DIR = 'models'

def evaluate_model(model, X_test, y_test, model_name):
    """Calculates and returns metrics for a given model."""
    predictions = model.predict(X_test)
    f1 = f1_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    return {"Model": model_name, "F1 Score": f1, "Precision": precision, "Recall": recall}

def evaluate_roberta(test_df):
    """Evaluates the fine-tuned RoBERTa model."""
    print("Evaluating RoBERTa model...")
    model_path = os.path.join(MODEL_DIR, 'roberta_model')
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    
    test_dataset = Dataset.from_pandas(test_df)
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=256)
        
    tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)
    
    trainer = Trainer(model=model)
    predictions = trainer.predict(tokenized_test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = test_df['label']
    
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return {"Model": "RoBERTa", "F1 Score": f1, "Precision": precision, "Recall": recall}

def evaluate_all(X_test_tfidf, X_test_stats, test_df):
    """Loads all models, evaluates them, and evaluates the ensemble."""
    print("\n--- Starting Evaluation Phase ---")
    y_test = test_df['label']
    
    # Load classic ML models
    lr_model = joblib.load(os.path.join(MODEL_DIR, 'logistic_regression.pkl'))
    xgb_model = joblib.load(os.path.join(MODEL_DIR, 'xgboost.pkl'))
    
    # Get predictions
    lr_preds = lr_model.predict(X_test_tfidf)
    xgb_preds = xgb_model.predict(X_test_stats)
    
    # Evaluate classic models
    lr_metrics = evaluate_model(lr_model, X_test_tfidf, y_test, "Logistic Regression")
    xgb_metrics = evaluate_model(xgb_model, X_test_stats, y_test, "XGBoost")
    
    # Evaluate RoBERTa
    roberta_metrics = evaluate_roberta(test_df)
    # Need RoBERTa preds for ensemble
    model_path = os.path.join(MODEL_DIR, 'roberta_model')
    roberta_model = RobertaForSequenceClassification.from_pretrained(model_path)
    roberta_tokenizer = RobertaTokenizer.from_pretrained(model_path)
    test_dataset = Dataset.from_pandas(test_df)
    tokenized_test_dataset = test_dataset.map(lambda e: roberta_tokenizer(e['text'], padding='max_length', truncation=True, max_length=256), batched=True)
    trainer = Trainer(model=roberta_model)
    roberta_raw_preds = trainer.predict(tokenized_test_dataset)
    roberta_preds = np.argmax(roberta_raw_preds.predictions, axis=1)
    
    # Ensemble with Voting
    print("Evaluating Ensemble model (Majority Vote)...")
    # Combine predictions: each column is a model's prediction
    all_preds = np.vstack([lr_preds, xgb_preds, roberta_preds]).T
    # The majority vote is the mode of each row
    ensemble_preds = pd.DataFrame(all_preds).mode(axis=1)[0].values
    
    ensemble_f1 = f1_score(y_test, ensemble_preds)
    ensemble_precision = precision_score(y_test, ensemble_preds)
    ensemble_recall = recall_score(y_test, ensemble_preds)
    
    ensemble_metrics = {
        "Model": "Ensemble (Voting)", 
        "F1 Score": ensemble_f1, 
        "Precision": ensemble_precision, 
        "Recall": ensemble_recall
    }
    
    # Compile and return results
    results_df = pd.DataFrame([lr_metrics, xgb_metrics, roberta_metrics, ensemble_metrics])
    return results_df
