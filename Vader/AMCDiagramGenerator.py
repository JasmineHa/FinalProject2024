import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

directory = 'D:/Msc/Final Project/Analogy Matreials Corpus AMC/Psychology Texts'
corpus = 'Human'
# directory = 'D:/Msc/Final Project/Analogy Matreials Corpus AMC/Llama2 Analogy Matreials Corpus'
# corpus = 'Llama2'

output_dir = 'D:/Msc/Final Project/Result Diagrams/Vader/PsychologyCorpus'
os.makedirs(output_dir, exist_ok=True)


def process_text_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='latin-1') as file:
                text = file.read()
        except UnicodeDecodeError as e:
            print(f"Error processing file {file_path}: {e}")
            return None, None

    sentiment = sia.polarity_scores(text)
    compound_score = sentiment['compound']
    return text, compound_score


texts = []
compound_scores = []

for file_name in os.listdir(directory):
    file = os.path.join(directory, file_name)
    text, compound_score = process_text_file(file)
    if text is not None and compound_score is not None:
        texts.append(text)
        compound_scores.append(compound_score)

if texts:
    df = pd.DataFrame({
        'text': texts,
        'compound': compound_scores
    })

    mean_sentiment = df['compound'].mean()
    std_sentiment = df['compound'].std()
    min_sentiment = df['compound'].min()
    max_sentiment = df['compound'].max()

    print(f"Mean Sentiment Score: {mean_sentiment}")
    print(f"Standard Deviation: {std_sentiment}")
    print(f"Minimum Sentiment Score: {min_sentiment}")
    print(f"Maximum Sentiment Score: {max_sentiment}")

    sns.set(style="whitegrid")
    fig, ax = plt.subplots(3, 1, figsize=(10, 18))

    # Histogram
    sns.histplot(df['compound'], color="blue", kde=False, bins=30, ax=ax[0], alpha=0.6)
    ax[0].set_title(corpus + ': Distribution of Sentiment Scores')
    ax[0].set_xlabel('Compound Sentiment Score')
    ax[0].set_ylabel('Frequency')

    # Boxplot
    box = sns.boxplot(y=df['compound'], color="blue", ax=ax[1])
    ax[1].set_title(corpus + ': Boxplot of Sentiment Scores')
    ax[1].set_ylabel('Compound Sentiment Score')
    for patch in box.patches:
        patch.set_alpha(0.6)
    for line in ax[1].lines:
        line.set_alpha(0.6)

    # Violin plot
    violin = sns.violinplot(y=df['compound'], inner="quartile", color="blue", ax=ax[2])
    ax[2].set_title(corpus + ': Violin Plot of Sentiment Scores')
    ax[2].set_ylabel('Compound Sentiment Score')
    for patch in violin.collections:
        patch.set_alpha(0.6)

    plt.tight_layout()

    combined_output_path = os.path.join(output_dir, f'{corpus}.png')
    fig.savefig(combined_output_path)

    plt.show()

else:
    print("Null")
