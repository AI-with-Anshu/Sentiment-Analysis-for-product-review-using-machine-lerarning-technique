"""
preprocessing.py
----------------
Text cleaning and TF-IDF feature extraction pipeline
for sentiment analysis on Amazon product reviews.
"""

import re
import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

STOP_WORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """
    Full text cleaning pipeline:
    - Lowercase
    - Remove URLs, HTML tags, special characters
    - Tokenize
    - Remove stopwords
    - Lemmatize
    """
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords and lemmatize
    tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token not in STOP_WORDS and len(token) > 2
    ]

    return ' '.join(tokens)


def load_and_preprocess(filepath: str) -> pd.DataFrame:
    """
    Load CSV dataset and apply full preprocessing pipeline.

    Expected CSV columns: 'review_text', 'sentiment'
    Sentiment labels: 'positive', 'negative', 'neutral'

    Returns:
        pd.DataFrame with original and cleaned text columns
    """
    print(f"Loading dataset from: {filepath}")
    df = pd.read_csv(filepath)

    print(f"Dataset shape: {df.shape}")
    print(f"Sentiment distribution:\n{df['sentiment'].value_counts()}\n")

    # Drop nulls
    df.dropna(subset=['review_text', 'sentiment'], inplace=True)

    # Apply cleaning
    print("Applying text preprocessing pipeline...")
    df['cleaned_text'] = df['review_text'].apply(clean_text)

    # Remove empty cleaned texts
    df = df[df['cleaned_text'].str.strip() != '']

    print(f"Dataset after cleaning: {df.shape}")
    return df


def build_tfidf_features(
    train_texts,
    test_texts=None,
    max_features: int = 10000,
    ngram_range: tuple = (1, 2)
):
    """
    Fit TF-IDF vectorizer on training data and transform train/test sets.

    Args:
        train_texts: Iterable of training text strings
        test_texts: Iterable of test text strings (optional)
        max_features: Maximum number of vocabulary features
        ngram_range: N-gram range for TF-IDF

    Returns:
        X_train_tfidf, X_test_tfidf (if test provided), fitted vectorizer
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,       # Apply log normalization to TF
        min_df=2,                # Ignore terms appearing in fewer than 2 docs
        max_df=0.95              # Ignore terms appearing in more than 95% of docs
    )

    print(f"Building TF-IDF features (max={max_features}, ngram={ngram_range})...")
    X_train = vectorizer.fit_transform(train_texts)
    print(f"TF-IDF feature matrix shape: {X_train.shape}")

    if test_texts is not None:
        X_test = vectorizer.transform(test_texts)
        return X_train, X_test, vectorizer

    return X_train, vectorizer
