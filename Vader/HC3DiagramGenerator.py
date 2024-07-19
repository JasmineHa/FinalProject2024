import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

# initialize the sentiment analyzer
sia = SentimentIntensityAnalyzer()

# path to the corpus
df = pd.read_csv('D:/Msc/Final Project/HC3 corpus/medicine.csv')
corpus = 'medicine'

# export the diagrams
output_dir = 'D:/Msc/Final Project/Result Diagrams/Vader/HC3'


def analyze_sentiment(answers):
    if isinstance(answers, str):
        return [sia.polarity_scores(answers)]
    elif isinstance(answers, list):
        return [sia.polarity_scores(answer) for answer in answers]
    return []


df['human_sentiments'] = df['human_answers'].apply(analyze_sentiment)
df['chatgpt_sentiments'] = df['chatgpt_answers'].apply(analyze_sentiment)

def mean_compound(sentiments):
    if sentiments:
        return pd.Series([x['compound'] for x in sentiments]).mean()
    return 0


df['mean_human_compound'] = df['human_sentiments'].apply(mean_compound)
df['mean_chatgpt_compound'] = df['chatgpt_sentiments'].apply(mean_compound)

# Calculation of Mean, Std Dev, Max and Min
stats_human = {
    'mean': df['mean_human_compound'].mean(),
    'std_dev': df['mean_human_compound'].std(),
    'max': df['mean_human_compound'].max(),
    'min': df['mean_human_compound'].min()
}

stats_chatgpt = {
    'mean': df['mean_chatgpt_compound'].mean(),
    'std_dev': df['mean_chatgpt_compound'].std(),
    'max': df['mean_chatgpt_compound'].max(),
    'min': df['mean_chatgpt_compound'].min()
}

print("Human Sentiments - Mean: {:.2f}, Std Dev: {:.2f}, Max: {:.2f}, Min: {:.2f}".format(
    stats_human['mean'], stats_human['std_dev'], stats_human['max'], stats_human['min']))
print("ChatGPT Sentiments - Mean: {:.2f}, Std Dev: {:.2f}, Max: {:.2f}, Min: {:.2f}".format(
    stats_chatgpt['mean'], stats_chatgpt['std_dev'], stats_chatgpt['max'], stats_chatgpt['min']))


# Histogram
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.histplot(df['mean_human_compound'], color="blue", label='Human', kde=False, bins=30, alpha=0.6)
sns.histplot(df['mean_chatgpt_compound'], color="red", label='ChatGPT', kde=False, bins=30, alpha=0.6)
plt.title(corpus + ': Distribution of Mean Sentiment Scores')
plt.xlabel('Mean Compound Sentiment Score')
plt.ylabel('Frequency')
plt.legend()
hist_output_path = os.path.join(output_dir, f'{corpus}_histogram.png')
plt.savefig(hist_output_path)
plt.show()
plt.close()

# Boxplot
plt.figure(figsize=(10, 6))
palette = ['blue', 'red']
box = sns.boxplot(data=df[['mean_human_compound', 'mean_chatgpt_compound']], palette=palette)
plt.title(corpus + ': Boxplot of Mean Sentiment Scores')
plt.ylabel('Compound Sentiment Score')
plt.xticks([0, 1], ['Human', 'ChatGPT'])
for patch in box.patches:
    patch.set_alpha(0.6)
for line in plt.gca().lines:
    line.set_alpha(0.6)
box_output_path = os.path.join(output_dir, f'{corpus}_boxplot.png')
plt.savefig(box_output_path)
plt.show()
plt.close()

# Violin plot
plt.figure(figsize=(10, 6))
violin = sns.violinplot(data=df[['mean_human_compound', 'mean_chatgpt_compound']], palette=palette)
plt.title(corpus + ': Violin Plot of Mean Sentiment Scores')
plt.ylabel('Compound Sentiment Score')
plt.xticks([0, 1], ['Human', 'ChatGPT'])
for patch in violin.collections:
    patch.set_alpha(0.6)
violin_output_path = os.path.join(output_dir, f'{corpus}_violinplot.png')
plt.savefig(violin_output_path)
plt.show()
plt.close()

