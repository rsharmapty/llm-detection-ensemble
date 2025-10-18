import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

def create_visualizations(df, save_dir='plots'):
    """
    Generates and saves visualizations for the dataset.

    Args:
        df (pd.DataFrame): The training dataframe containing 'text' and 'label' columns.
        save_dir (str): Directory to save the plots.
    """
    print("Generating data visualizations...")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Separate human and AI text
    human_text = " ".join(df[df['label'] == 0]['text'])
    ai_text = " ".join(df[df['label'] == 1]['text'])

    # 1. Text Length Distribution
    df['text_length'] = df['text'].str.len()
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='text_length', hue='label', bins=50, kde=True, palette=['#4c72b0', '#dd8452'])
    plt.title('Distribution of Text Length for Human vs. AI Generated Text')
    plt.xlabel('Text Length (Number of Characters)')
    plt.ylabel('Frequency')
    plt.legend(title='Source', labels=['AI Generated', 'Human Written'])
    plt.savefig(os.path.join(save_dir, 'text_length_distribution.png'))
    plt.close()
    print(f"Saved text length distribution plot to {save_dir}/")

    # 2. Word Clouds
    # Human Word Cloud
    wordcloud_human = WordCloud(width=800, height=400, background_color='white').generate(human_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud_human, interpolation='bilinear')
    plt.axis('off')
    plt.title('Most Common Words in Human-Written Text')
    plt.savefig(os.path.join(save_dir, 'wordcloud_human.png'))
    plt.close()
    print(f"Saved human word cloud to {save_dir}/")
    
    # AI Word Cloud
    wordcloud_ai = WordCloud(width=800, height=400, background_color='black', colormap='viridis').generate(ai_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud_ai, interpolation='bilinear')
    plt.axis('off')
    plt.title('Most Common Words in AI-Generated Text')
    plt.savefig(os.path.join(save_dir, 'wordcloud_ai.png'))
    plt.close()
    print(f"Saved AI word cloud to {save_dir}/")

if __name__ == '__main__':
    # For testing the script directly
    from data_loader import load_and_prepare_data
    train_df, _ = load_and_prepare_data(sample_size=100)
    create_visualizations(train_df)
