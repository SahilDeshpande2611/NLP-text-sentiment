# NLP Text Sentiment Analysis

This project performs sentiment and emotion classification on text data using Natural Language Processing (NLP) techniques and machine learning models.

## Overview

- **Dataset:** The [Emotions dataset for NLP](https://www.kaggle.com/datasets) from Kaggle (`train.txt`) contains text samples labeled with emotions (e.g., sadness, anger, joy, love, etc.).
- **Goal:** Predict the emotion of a given text using machine learning models.

## Workflow

1. **Data Loading:**  
   Load the dataset from `train.txt` (semicolon-separated).

2. **Exploratory Data Analysis:**  
   - Check for missing values.
   - Explore unique emotion labels.

3. **Preprocessing:**  
   - Convert text to lowercase.
   - Remove punctuation, numbers, emojis, and special characters.
   - Remove stopwords.
   - Tokenize and clean text.

4. **Feature Extraction:**  
   - Bag of Words (CountVectorizer)
   - TF-IDF (TfidfVectorizer)

5. **Model Training:**  
   - Naive Bayes classifier
   - Logistic Regression

6. **Evaluation:**  
   - Accuracy score on test data.

## Usage

1. **Requirements:**
   - Python 3.x
   - pandas, numpy, matplotlib, seaborn
   - scikit-learn
   - nltk

2. **Run the Notebook:**
   - Open `proj.ipynb` in Jupyter Notebook or VS Code.
   - Execute cells sequentially to preprocess data, train models, and evaluate results.

3. **Dataset:**
   - Ensure `train.txt` is present in the project directory.

## Notes

- The notebook includes all preprocessing and modeling steps.
- You can modify the notebook to experiment with other models or preprocessing techniques.

## Example

```python
import pandas as pd
df = pd.read_csv('train.txt', sep=';', header=None, names=['text', 'emotion'])
# ...preprocessing and modeling...
```

---

**Author:**  
Sahil Deshpande  
[LinkedIn](https://www.linkedin.com/in/sahilsdeshpande)
