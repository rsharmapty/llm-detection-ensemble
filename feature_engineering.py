import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import pandas as pd

def create_tfidf_features(train_df, test_df):
    """Creates TF-IDF features from text data."""
    print("Creating TF-IDF features...")
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_tfidf = tfidf_vectorizer.fit_transform(train_df['text'])
    X_test_tfidf = tfidf_vectorizer.transform(test_df['text'])
    return X_train_tfidf, X_test_tfidf, tfidf_vectorizer

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

def calculate_perplexity(text):
    """Calculates perplexity of a text using GPT-2 as a surrogate model."""
    if gpt2_model is None:
        return 0
    try:
        inputs = gpt2_tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
        with torch.no_grad():
            outputs = gpt2_model(**inputs, labels=inputs["input_ids"])
            log_likelihood = outputs.loss
        ppl = torch.exp(log_likelihood)
        return ppl.item()
    except Exception:
        return 0 # Return 0 if text is too long or causes an error

def extract_statistical_features(df):
    """Extracts a set of statistical and linguistic features from text."""
    print("Extracting statistical features...")
    features = {}
    # Basic features
    features['text_length'] = df['text'].apply(len)
    features['word_count'] = df['text'].apply(lambda x: len(x.split()))
    features['sentence_count'] = df['text'].apply(lambda x: len(nltk.sent_tokenize(x)))
    
    # Complexity features
    features['avg_word_length'] = features['text_length'] / features['word_count']
    features['avg_sentence_length'] = features['word_count'] / features['sentence_count']
    
    # Perplexity (this can be slow)
    print("Calculating perplexity for each text... (This may take a while)")
    features['perplexity'] = df['text'].apply(calculate_perplexity)
    
    return pd.DataFrame(features).fillna(0)


def create_statistical_features(train_df, test_df):
    """Creates statistical feature sets for training and testing."""
    X_train_stats = extract_statistical_features(train_df)
    X_test_stats = extract_statistical_features(test_df)
    return X_train_stats, X_test_stats

if __name__ == '__main__':
    # For testing the script directly
    from data_loader import load_and_prepare_data
    import nltk
    nltk.download('punkt')
    train, test = load_and_prepare_data(sample_size=10)
    
    print("\n--- Testing TF-IDF ---")
    X_train_tf, X_test_tf, _ = create_tfidf_features(train, test)
    print(f"Train TF-IDF shape: {X_train_tf.shape}")
    print(f"Test TF-IDF shape: {X_test_tf.shape}")

    print("\n--- Testing Statistical Features ---")
    X_train_st, X_test_st = create_statistical_features(train, test)
    print("Train Statistical Features:")
    print(X_train_st.head())
