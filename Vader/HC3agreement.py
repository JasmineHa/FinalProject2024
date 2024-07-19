import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# This file calculates the agreement level of human and ChatGPT sentiment scores by Vader

# Load the .jsonl file into a DataFrame
df = pd.read_json('C:/Users/Kanako/Downloads/HC3 corpus/wiki_csai.jsonl', lines=True)

# Download the VADER lexicon
nltk.download('vader_lexicon')

# Initialize the sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Define a function to analyze sentiment of each answer
def analyze_sentiment(answers):
    # This assumes answers are either a list of strings or a single string
    if isinstance(answers, list):
        return [sia.polarity_scores(answer) for answer in answers]
    return sia.polarity_scores(answers)

# Apply sentiment analysis to human and ChatGPT answers
df['human_sentiments'] = df['human_answers'].apply(analyze_sentiment)
df['chatgpt_sentiments'] = df['chatgpt_answers'].apply(analyze_sentiment)

print(df[['human_sentiments', 'chatgpt_sentiments']])

# Define a function to extract the compound score from sentiment analysis results
def extract_compound_score(sentiments):
    if isinstance(sentiments, list):
        return [sentiment['compound'] for sentiment in sentiments]
    return sentiments['compound']

# Extract compound scores
df['human_compound'] = df['human_sentiments'].apply(extract_compound_score)
df['chatgpt_compound'] = df['chatgpt_sentiments'].apply(extract_compound_score)

# Define a function to calculate the average compound score
def average_compound_score(compound_scores):
    if isinstance(compound_scores, list):
        return sum(compound_scores) / len(compound_scores)
    return compound_scores

# Calculate average compound scores
df['human_avg_compound'] = df['human_compound'].apply(average_compound_score)
df['chatgpt_avg_compound'] = df['chatgpt_compound'].apply(average_compound_score)

def categorize_difference(human, ai):
    difference = abs(human - ai)
    if difference <= 0.1:
        return 'Agreement'
    elif difference <= 0.5:
        return 'Mild Disagreement'
    else:
        return 'Strong Disagreement'

# Apply categorization
df['agreement_level'] = df.apply(lambda x: categorize_difference(x['human_avg_compound'], x['chatgpt_avg_compound']), axis=1)

agreement_counts = df['agreement_level'].value_counts()
print(agreement_counts)




