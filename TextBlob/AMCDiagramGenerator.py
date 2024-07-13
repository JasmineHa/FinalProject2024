import os
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns


# Llama2 corpus
# directory = 'D:/Msc/Final Project/Analogy Matreials Corpus AMC/Llama2 Analogy Matreials Corpus'
# corpus = 'Llama2'

# Human corpus
directory = 'D:/Msc/Final Project/Analogy Matreials Corpus AMC/Psychology Texts'
corpus = 'human'

output_dir = 'D:/Msc/Final Project/Analysis Results'
os.makedirs(output_dir, exist_ok=True)

# Function for TextBlob SA
def compute_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity, analysis.sentiment.subjectivity


def process_txt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            polarity, subjectivity = compute_sentiment(text)
            return {'file': os.path.basename(file_path), 'text': text, 'polarity': polarity, 'subjectivity': subjectivity}
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='latin1') as file:
                text = file.read()
                polarity, subjectivity = compute_sentiment(text)
                return {'file': os.path.basename(file_path), 'text': text, 'polarity': polarity, 'subjectivity': subjectivity}
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return None


# process all text files in the directory
all_data = []
for file_name in os.listdir(directory):
    if file_name.endswith('.txt'):
        print(f"Processing file: {file_name}")  # for debug
        full_path = os.path.join(directory, file_name)
        processed_data = process_txt(full_path)
        if processed_data is not None:
            all_data.append(processed_data)

# Combine all processed data into a DataFrame
if all_data:
    combined_df = pd.DataFrame(all_data)

    # Extract polarity and subjectivity scores
    polarity_scores = combined_df['polarity']
    subjectivity_scores = combined_df['subjectivity']

    # polarity
    mean_polarity = polarity_scores.mean()
    std_polarity = polarity_scores.std()
    min_polarity = polarity_scores.min()
    max_polarity = polarity_scores.max()

    # subjectivity
    mean_subjectivity = subjectivity_scores.mean()
    std_subjectivity = subjectivity_scores.std()
    min_subjectivity = subjectivity_scores.min()
    max_subjectivity = subjectivity_scores.max()


    print(f"Polarity - Mean: {mean_polarity}, Std Dev: {std_polarity}, Min: {min_polarity}, Max: {max_polarity}")
    print(f"Subjectivity - Mean: {mean_subjectivity}, Std Dev: {std_subjectivity}, Min: {min_subjectivity}, Max: {max_subjectivity}")

    # set the same range for all plots
    polarity_range = (-1, 1)
    subjectivity_range = (0, 1)

    # a combined DataFrame for plotting
    plot_df = pd.DataFrame({
        'Polarity': polarity_scores,
        'Subjectivity': subjectivity_scores
    })

    # boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=plot_df, palette=['pink', 'MediumTurquoise'])
    plt.title(f'{corpus}: Boxplot of Polarity and Subjectivity Scores')
    plt.ylabel('Score')
    plt.xlabel('Metric')
    plt.ylim(polarity_range)  # Ensure the same y-axis range
    output_boxplot_path = os.path.join(output_dir, f'{corpus}_boxplot.png')
    plt.savefig(output_boxplot_path)
    plt.show()

    # violin plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=plot_df, palette=['pink', 'MediumTurquoise'])
    plt.title(f'{corpus}: Violin Plot of Polarity and Subjectivity Scores')
    plt.ylabel('Score')
    plt.xlabel('Metric')
    plt.ylim(polarity_range)  # Ensure the same y-axis range
    output_violinplot_path = os.path.join(output_dir, f'{corpus}_violinplot.png')
    plt.savefig(output_violinplot_path)
    plt.show()

    # polarity histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(polarity_scores, kde=False, color="pink", bins=30)
    plt.title(f'{corpus}: Histogram of Polarity Scores')
    plt.xlabel('Polarity Score')
    plt.ylabel('Frequency')
    plt.xlim(polarity_range)  # Ensure the same x-axis range
    output_polarity_hist_path = os.path.join(output_dir, f'{corpus}_polarity_histogram.png')
    plt.savefig(output_polarity_hist_path)
    plt.show()

    # subjectivity histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(subjectivity_scores, kde=False, color="MediumTurquoise", bins=30)
    plt.title(f'{corpus}: Histogram of Subjectivity Scores')
    plt.xlabel('Subjectivity Score')
    plt.ylabel('Frequency')
    plt.xlim(subjectivity_range)  # Ensure the same x-axis range
    output_subjectivity_hist_path = os.path.join(output_dir, f'{corpus}_subjectivity_histogram.png')
    plt.savefig(output_subjectivity_hist_path)
    plt.show()

else:
    print("Null")
