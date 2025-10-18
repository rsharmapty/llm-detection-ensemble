import pandas as pd
from datasets import load_dataset

def load_and_prepare_data(sample_size=2000):
    """
    Downloads the HC3 dataset, balances it, and splits it into training and testing sets.
    
    Args:
        sample_size (int): The number of samples for each class (human, AI) to use.
                           Helps in managing memory and training time for demonstration.

    Returns:
        tuple: A tuple containing training and testing pandas DataFrames.
               (train_df, test_df)
    """
    print("Loading HC3 dataset from Hugging Face...")
    # Using the 'all' configuration to get a mix of sources
    dataset = load_dataset("Hello-SimpleAI/HC3", "all", split='train')
    
    # Convert to pandas DataFrame for easier manipulation
    df = dataset.to_pandas()
    
    # Separate human and AI answers
    human_df = df[['question', 'human_answers']].rename(columns={'human_answers': 'text'})
    human_df['text'] = human_df['text'].str.join(' ')
    human_df['label'] = 0 # 0 for human

    chatgpt_df = df[['question', 'chatgpt_answers']].rename(columns={'chatgpt_answers': 'text'})
    chatgpt_df['text'] = chatgpt_df['text'].str.join(' ')
    chatgpt_df['label'] = 1 # 1 for AI
    
    # Remove rows with empty text
    human_df = human_df[human_df['text'].str.len() > 0]
    chatgpt_df = chatgpt_df[chatgpt_df['text'].str.len() > 0]

    print(f"Original human samples: {len(human_df)}")
    print(f"Original AI samples: {len(chatgpt_df)}")
    
    # Balance the dataset by taking a smaller sample
    human_sample = human_df.sample(n=sample_size, random_state=42)
    chatgpt_sample = chatgpt_df.sample(n=sample_size, random_state=42)
    
    # Combine and shuffle the data
    balanced_df = pd.concat([human_sample, chatgpt_sample]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Using a balanced subset of {len(balanced_df)} samples.")
    
    # Split into training and testing sets (80/20 split)
    train_size = int(0.8 * len(balanced_df))
    train_df = balanced_df[:train_size]
    test_df = balanced_df[train_size:]
    
    print(f"Training set size: {len(train_df)}")
    print(f"Testing set size: {len(test_df)}")
    
    return train_df, test_df

if __name__ == '__main__':
    # For testing the script directly
    train, test = load_and_prepare_data(sample_size=100)
    print("\nTraining Data Head:")
    print(train.head())
    print("\nTesting Data Head:")
    print(test.head())
