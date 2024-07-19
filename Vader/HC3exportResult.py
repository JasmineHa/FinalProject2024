import os
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# This code exports a csv file of the corpus with the sentiment scores
# to be able to access the content and sort the data by the scores in Excel

nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

df = pd.read_csv('D:/Msc/Final Project/HC3 corpus/finance.csv')
corpus = 'finance'

def analyze_sentiment(answers):
    if isinstance(answers, str):
        scores = sia.polarity_scores(answers)
        return scores
    return None


df['human_sentiment_scores'] = df['human_answers'].apply(analyze_sentiment)
df['chatgpt_sentiment_scores'] = df['chatgpt_answers'].apply(analyze_sentiment)

df['human_negative'] = df['human_sentiment_scores'].apply(lambda x: x['neg'])
df['human_neutral'] = df['human_sentiment_scores'].apply(lambda x: x['neu'])
df['human_positive'] = df['human_sentiment_scores'].apply(lambda x: x['pos'])
df['human_compound'] = df['human_sentiment_scores'].apply(lambda x: x['compound'])

df['chatgpt_negative'] = df['chatgpt_sentiment_scores'].apply(lambda x: x['neg'])
df['chatgpt_neutral'] = df['chatgpt_sentiment_scores'].apply(lambda x: x['neu'])
df['chatgpt_positive'] = df['chatgpt_sentiment_scores'].apply(lambda x: x['pos'])
df['chatgpt_compound'] = df['chatgpt_sentiment_scores'].apply(lambda x: x['compound'])

output_dir = 'D:/Msc/Final Project/HC3 corpus/withResults'
output_file_path = os.path.join(output_dir, corpus + '_with_detailed_sentiment_scores.csv')

df.to_csv(output_file_path, index=False)

