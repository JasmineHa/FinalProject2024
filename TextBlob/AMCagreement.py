import os
from textblob import TextBlob

# This file calculates the agreement level of human and Llama2 stories sentiment scores by TextBlob


# Define directories for human and AI-generated text files
human_dir = 'D:/Msc/Final Project/Analogy Matreials Corpus AMC/Psychology Texts'
llama2_dir = 'D:/Msc/Final Project/Analogy Matreials Corpus AMC/Llama2 Analogy Matreials Corpus'

def analyze_files_in_directory(directory):
    compound_scores = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                    blob = TextBlob(text)
                    # TextBlob returns a namedtuple of form (polarity, subjectivity)
                    # Polarity is a float within the range [-1.0, 1.0]
                    compound_scores.append(blob.sentiment.polarity)
            except UnicodeDecodeError:
                print(f"Failed to decode {filename}, skipped.")
    return compound_scores

# Analyze sentiment scores for both human and AI directories
human_compounds = analyze_files_in_directory(human_dir)
ai_compounds = analyze_files_in_directory(llama2_dir)

# Categorize agreement level based on compound scores
def categorize_difference(human, ai):
    difference = abs(human - ai)
    if difference <= 0.1:
        return 'Agreement'
    elif difference <= 0.5:
        return 'Mild Disagreement'
    else:
        return 'Strong Disagreement'

# Compare the human and AI scores and categorize
agreement_levels = []
for h_score, a_score in zip(human_compounds, ai_compounds):
    agreement = categorize_difference(h_score, a_score)
    agreement_levels.append(agreement)

# Count agreement levels
agreement_counts = {}
for level in agreement_levels:
    if level in agreement_counts:
        agreement_counts[level] += 1
    else:
        agreement_counts[level] = 1

# Print the agreement level counts
print("Agreement Level Counts:")
for level, count in agreement_counts.items():
    print(f"{level}: {count}")
