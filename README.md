# NLP Text Sentiment Analysis

This project performs sentiment and emotion classification on text data using Natural Language Processing (NLP) techniques and machine learning models.

## Overview

- **Dataset:** The [Emotions dataset for NLP](https://www.kaggle.com/datasets) from Kaggle (`train.txt`) contains text samples labeled with emotions (e.g., sadness, anger, joy, love, etc.).
- **Goal:** Predict the emotion of a given text using machine learning models.
- **Web App:** A Streamlit app (`app.py`) provides an interactive interface for emotion prediction.

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

7. **Saving Artifacts:**  
   - Trained model (`regression_sentimentanalysis.pkl`)
   - TF-IDF vectorizer (`scaler.pkl`)
   - Label encoder (`label_encoder.pkl`)
   - Emotion label mapping (`emotion_numbers.pkl`)

## Usage

### Requirements

- Python 3.x
- pandas, numpy, matplotlib, seaborn
- scikit-learn
- nltk
- streamlit
- joblib

### Run the Notebook

- Open `proj.ipynb` in Jupyter Notebook or VS Code.
- Execute cells sequentially to preprocess data, train models, and evaluate results.

### Run the Web App

1. Make sure all required `.pkl` files are present in the project directory.
2. Run the following command in your terminal:
   ```
   streamlit run app.py
   ```
3. Open your browser and go to:
   ```
   http://localhost:8501/
   ```
   You will see the Sentiment Analysis interface. Enter a sentence to get the predicted emotion.

### Dataset

- Ensure `train.txt` is present in the project directory.

## Notes

- The notebook includes all preprocessing and modeling steps.
- The Streamlit app displays the emotion label mapping using `emotion_numbers.pkl`.
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

### Connect :
- **LinkedIn**: [Connect with me professionally](https://in.linkedin.com/in/sahilsdeshpande)

Thank you for your support, and I look forward to connecting with you!
