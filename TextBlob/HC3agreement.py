import pandas as pd
from textblob import TextBlob

# This file calculates the agreement level of human and ChatGPT sentiment scores by TextBlob

# Path to the corpus file
df = pd.read_csv('D:/Msc/Final Project/HC3 corpus/medicine.csv')
corpus = "medicine "

# Function to apply sentiment analysis using TextBlob
def analyze_sentiment(answer):
    return TextBlob(answer).sentiment.polarity

# Apply sentiment analysis to human and ChatGPT answers directly within the DataFrame
df['human_polarity'] = df['human_answers'].apply(analyze_sentiment)
df['chatgpt_polarity'] = df['chatgpt_answers'].apply(analyze_sentiment)

# Function to categorize agreement level based on the polarity scores
def categorize_difference(human_polarity, chatgpt_polarity):
    difference = abs(human_polarity - chatgpt_polarity)
    if difference <= 0.1:
        return 'Agreement'
    elif difference <= 0.5:
        return 'Mild Disagreement'
    else:
        return 'Strong Disagreement'

# Calculate the agreement level based on polarity scores
df['agreement_level'] = df.apply(lambda x: categorize_difference(x['human_polarity'], x['chatgpt_polarity']), axis=1)

# Print the resulting DataFrame with sentiments and agreement levels
print(df[['human_answers', 'chatgpt_answers', 'human_polarity', 'chatgpt_polarity', 'agreement_level']])

# Count and print the agreement levels
agreement_counts = df['agreement_level'].value_counts()
print(corpus + "Agreement Level Counts:")
print(agreement_counts)
