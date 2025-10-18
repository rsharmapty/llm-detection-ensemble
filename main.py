import pandas as pd
from data_loader import load_and_prepare_data
from visualization import create_visualizations
from feature_engineering import create_tfidf_features, create_statistical_features
from models import train_logistic_regression, train_xgboost, train_roberta
from evaluate import evaluate_all

def main():
    """
    Main function to run the entire LLM detection pipeline.
    """
    print("--- LLM Generated Text Detection Project ---")
    
    # --- 1. Data Loading and Preprocessing ---
    # Using a smaller sample size for quicker execution in this demo project.
    # Increase sample_size for a more robust evaluation.
    train_df, test_df = load_and_prepare_data(sample_size=2500)
    
    # --- 2. Data Visualization ---
    # Running visualization on the training data
    create_visualizations(train_df.copy())
    
    # --- 3. Feature Engineering ---
    # For Logistic Regression
    X_train_tfidf, X_test_tfidf, _ = create_tfidf_features(train_df, test_df)
    
    # For XGBoost
    X_train_stats, X_test_stats = create_statistical_features(train_df, test_df)
    
    y_train = train_df['label']
    
    # --- 4. Model Training ---
    # Train and save the three individual models
    train_logistic_regression(X_train_tfidf, y_train)
    train_xgboost(X_train_stats, y_train)
    train_roberta(train_df[['text', 'label']]) # RoBERTa needs text and labels
    
    # --- 5. Evaluation ---
    # Evaluate all models and the ensemble
    results_df = evaluate_all(X_test_tfidf, X_test_stats, test_df)
    
    # --- 6. Final Report ---
    print("\n\n--- FINAL PERFORMANCE REPORT ---")
    print(results_df.to_string(index=False))
    print("\n--- Project Execution Finished ---")

if __name__ == '__main__':
    main()
