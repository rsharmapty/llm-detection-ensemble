import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import pandas as pd

# Load a small surrogate model for perplexity calculation
try:
    print("Loading surrogate GPT-2 model for perplexity calculation...")
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
    gpt2_model.eval()
    print("GPT-2 model loaded successfully.")
except Exception as e:
    print(f"Could not load GPT-2 model. Perplexity feature will not be available. Error: {e}")
    gpt2_tokenizer = None
    gpt2_model = None

def create_tfidf_features(train_df, test_df, vectorizer_path='models/tfidf_vectorizer.pkl'):
    """Creates TF-IDF features from text data and saves the vectorizer."""
    print("Creating TF-IDF features...")
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_tfidf = tfidf_vectorizer.fit_transform(train_df['text'])
    X_test_tfidf = tfidf_vectorizer.transform(test_df['text'])
    
    # Save the vectorizer for inference
    import joblib
    import os
    if not os.path.exists('models'):
        os.makedirs('models')
    joblib.dump(tfidf_vectorizer, vectorizer_path)
    print(f"TF-IDF vectorizer saved to {vectorizer_path}")

    return X_train_tfidf, X_test_tfidf, tfidf_vectorizer

def calculate_perplexity(text):
    """Calculates perplexity of a text using GPT-2 as a surrogate model."""
    if gpt2_model is None or not text.strip():
        return 0
    try:
        inputs = gpt2_tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
        with torch.no_grad():
            outputs = gpt2_model(**inputs, labels=inputs["input_ids"])
            # The loss is the average negative log-likelihood
            log_likelihood = outputs.loss.item()
        ppl = np.exp(log_likelihood)
        return ppl
    except Exception:
        return 0 # Return 0 for errors

def extract_statistical_features(df):
    """Extracts a set of statistical and linguistic features from a DataFrame."""
    print("Extracting statistical features...")
    features = {}
    features['text_length'] = df['text'].apply(len)
    features['word_count'] = df['text'].apply(lambda x: len(x.split()))
    features['sentence_count'] = df['text'].apply(lambda x: len(nltk.sent_tokenize(x)))
    
    # Avoid division by zero for empty strings
    features['avg_word_length'] = (df['text'].str.len() / df['text'].str.split().str.len()).fillna(0)
    features['avg_sentence_length'] = (df['text'].str.split().str.len() / df['text'].str.split('[.!?]').str.len()).fillna(0)
    
    print("Calculating perplexity for each text... (This may take a while)")
    features['perplexity'] = df['text'].progress_apply(calculate_perplexity) if 'progress_apply' in dir(df['text']) else df['text'].apply(calculate_perplexity)
    
    return pd.DataFrame(features).fillna(0)

def create_statistical_features(train_df, test_df):
    """Creates statistical feature sets for training and testing DataFrames."""
    X_train_stats = extract_statistical_features(train_df)
    X_test_stats = extract_statistical_features(test_df)
    return X_train_stats, X_test_stats

def create_statistical_features_for_single_text(text):
    """Extracts statistical features for a single string of text."""
    data = {'text': [text]}
    df = pd.DataFrame(data)
    # Reuse the same extraction logic
    features_df = extract_statistical_features(df)
    return features_df

if __name__ == '__main__':
    from data_loader import load_and_prepare_data
    import nltk
    nltk.download('punkt')
    train, test = load_and_prepare_data(sample_size=10)
    
    print("\n--- Testing Single Text Feature Extraction ---")
    single_text_features = create_statistical_features_for_single_text("This is a test sentence.")
    print(single_text_features)

