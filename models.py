import os
import joblib
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

MODEL_DIR = 'models'

def train_logistic_regression(X_train, y_train):
    """Trains a Logistic Regression model and saves it."""
    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    joblib.dump(model, os.path.join(MODEL_DIR, 'logistic_regression.pkl'))
    print(f"Logistic Regression model saved to {MODEL_DIR}/")
    return model

def train_xgboost(X_train, y_train):
    """Trains an XGBoost model and saves it."""
    print("Training XGBoost model...")
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    joblib.dump(model, os.path.join(MODEL_DIR, 'xgboost.pkl'))
    print(f"XGBoost model saved to {MODEL_DIR}/")
    return model
    
def train_roberta(train_df):
    """Fine-tunes a RoBERTa model and saves it."""
    print("Training RoBERTa model... (This will take a significant amount of time)")
    
    # Convert pandas DataFrame to Hugging Face Dataset
    train_dataset = Dataset.from_pandas(train_df)
    
    # Initialize Tokenizer and Model
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
    
    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=256)

    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    
    # Define Training Arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(MODEL_DIR, 'roberta_results'),
        # Request 1: Increased epochs for more robust training
        num_train_epochs=3,
        per_device_train_batch_size=8,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        report_to="none" # Disable wandb/tensorboard logging for simplicity
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
    )

    # Train the model
    trainer.train()
    
    # Save the model
    trainer.save_model(os.path.join(MODEL_DIR, 'roberta_model'))
    tokenizer.save_pretrained(os.path.join(MODEL_DIR, 'roberta_model'))
    print(f"RoBERTa model saved to {MODEL_DIR}/roberta_model")
    return trainer.model, tokenizer

