# FinalProject: Sentiment Analysis of AI vs Human Responses

**Python version:** 3.9.13
**VADER Sentiment Analyzer:** 3.3.2
**TextBlob:** 0.18.0.post0

---

## Project Overview

This project performs sentiment analysis on the HC3 corpus, which contains paired human and AI-generated responses. The goal is to compare sentiment patterns between human and AI texts to explore differences in tone and emotional expression.

---

## Features

* Runs sentiment analysis using both **VADER** and **TextBlob** libraries.
* Generates insightful visualizations including:

  * Histogram
  * Box plot
  * Violin plot
* Automatically exports all generated diagrams to the `output` directory with clear, descriptive file names.

---

## Dataset

The analysis is performed on the HC3 corpus, publicly available here:
[HC3 Corpus on Hugging Face](https://huggingface.co/datasets/Hello-SimpleAI/HC3)

---

## How to Run

1. Ensure Python 3.9.13 is installed.
2. Install dependencies with:

   ```bash
   pip install vaderSentiment==3.3.2 textblob==0.18.0.post0 matplotlib seaborn pandas
   ```
3. Run the diagram generator scripts located in the repository. The scripts will read the corpus, perform sentiment analysis, and save the generated plots in the `output` folder.

---

## Output

You will find visualized diagrams that illustrate the distribution and variance of sentiment scores across human and AI-generated texts, facilitating easy comparison and discovery.
