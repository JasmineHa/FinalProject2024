import os
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('D:/Msc/Final Project/HC3 corpus/finance.csv')
corpus = "finance"

output_dir = 'D:/Msc/Final Project/Result Diagrams/TextBlob/HC3'
os.makedirs(output_dir, exist_ok=True)

# function for SA
def analyze_sentiment(answers):
    if isinstance(answers, list):
        return [{'polarity': TextBlob(answer).sentiment.polarity,
                 'subjectivity': TextBlob(answer).sentiment.subjectivity} for answer in answers]
    return {'polarity': 0, 'subjectivity': 0}


df['human_sentiments'] = df['human_answers'].apply(analyze_sentiment)
df['chatgpt_sentiments'] = df['chatgpt_answers'].apply(analyze_sentiment)


def mean_polarity(sentiments):
    return pd.Series([x['polarity'] for x in sentiments]).mean()


def mean_subjectivity(sentiments):
    return pd.Series([x['subjectivity'] for x in sentiments]).mean()


df['mean_human_polarity'] = df['human_sentiments'].apply(mean_polarity)
df['mean_chatgpt_polarity'] = df['chatgpt_sentiments'].apply(mean_polarity)
df['mean_human_subjectivity'] = df['human_sentiments'].apply(mean_subjectivity)
df['mean_chatgpt_subjectivity'] = df['chatgpt_sentiments'].apply(mean_subjectivity)



# Calculation of Mean, Std Dev, Max and Min
mean_human_polarity = df['mean_human_polarity'].mean()
std_human_polarity = df['mean_human_polarity'].std()
max_human_polarity = df['mean_human_polarity'].max()
min_human_polarity = df['mean_human_polarity'].min()

mean_chatgpt_polarity = df['mean_chatgpt_polarity'].mean()
std_chatgpt_polarity = df['mean_chatgpt_polarity'].std()
max_chatgpt_polarity = df['mean_chatgpt_polarity'].max()
min_chatgpt_polarity = df['mean_chatgpt_polarity'].min()

mean_human_subjectivity = df['mean_human_subjectivity'].mean()
std_human_subjectivity = df['mean_human_subjectivity'].std()
max_human_subjectivity = df['mean_human_subjectivity'].max()
min_human_subjectivity = df['mean_human_subjectivity'].min()

mean_chatgpt_subjectivity = df['mean_chatgpt_subjectivity'].mean()
std_chatgpt_subjectivity = df['mean_chatgpt_subjectivity'].std()
max_chatgpt_subjectivity = df['mean_chatgpt_subjectivity'].max()
min_chatgpt_subjectivity = df['mean_chatgpt_subjectivity'].min()


print(f"Human Polarity - Mean: {mean_human_polarity}, Std Dev: {std_human_polarity}, Max: {max_human_polarity}, Min: {min_human_polarity}")
print(f"ChatGPT Polarity - Mean: {mean_chatgpt_polarity}, Std Dev: {std_chatgpt_polarity}, Max: {max_chatgpt_polarity}, Min: {min_chatgpt_polarity}")

print(f"Human Subjectivity - Mean: {mean_human_subjectivity}, Std Dev: {std_human_subjectivity}, Max: {max_human_subjectivity}, Min: {min_human_subjectivity}")
print(f"ChatGPT Subjectivity - Mean: {mean_chatgpt_subjectivity}, Std Dev: {std_chatgpt_subjectivity}, Max: {max_chatgpt_subjectivity}, Min: {min_chatgpt_subjectivity}")

# Set the same range for all plots
x_range = (-1, 1)

# histograms
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(8, 6))
sns.histplot(df['mean_human_polarity'], color="blue", label='Human', kde=False, bins=30, ax=ax, alpha=0.6)
sns.histplot(df['mean_chatgpt_polarity'], color="red", label='ChatGPT', kde=False, bins=30, ax=ax, alpha=0.6)
ax.set_title(corpus + ': Distribution of Mean Polarity Scores')
ax.set_xlabel('Mean Polarity Score')
ax.set_ylabel('Frequency')
ax.set_xlim(x_range)
ax.legend()
hist_output_path = os.path.join(output_dir, f'{corpus}_histogram.png')
plt.savefig(hist_output_path)
plt.show()

# Generate boxplot and violin plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# boxplot
box_human = sns.boxplot(data=df, y='mean_human_subjectivity', ax=axes[1, 0], color="blue")
axes[1, 0].set_title(corpus + ': Box Plot of Human Subjectivity')
axes[1, 0].set_ylim(0, 1)
for patch in box_human.patches:
    patch.set_alpha(0.6)
for line in axes[1, 0].lines:
    line.set_alpha(0.6)

box_chatgpt = sns.boxplot(data=df, y='mean_chatgpt_subjectivity', ax=axes[1, 1], color="red")
axes[1, 1].set_title(corpus + ': Box Plot of ChatGPT Subjectivity')
axes[1, 1].set_ylim(0, 1)
for patch in box_chatgpt.patches:
    patch.set_alpha(0.6)
for line in axes[1, 1].lines:
    line.set_alpha(0.6)

# violin plot
violin_human = sns.violinplot(data=df, y='mean_human_polarity', ax=axes[0, 0], color="blue")
axes[0, 0].set_title(corpus + ': Violin Plot of Human Polarity')
axes[0, 0].set_ylim(x_range)
for patch in violin_human.collections:
    patch.set_alpha(0.6)

violin_chatgpt = sns.violinplot(data=df, y='mean_chatgpt_polarity', ax=axes[0, 1], color="red")
axes[0, 1].set_title(corpus + ': Violin Plot of ChatGPT Polarity')
axes[0, 1].set_ylim(x_range)
for patch in violin_chatgpt.collections:
    patch.set_alpha(0.6)

# Add a large title to the whole figure
fig.suptitle(corpus, fontsize=20)

plt.tight_layout()
violin_box_output_path = os.path.join(output_dir, f'{corpus}_violin_box_plots.png')
plt.savefig(violin_box_output_path)
plt.show()
