"""
predict.py
----------
Load saved SVM model and predict sentiment on new text input.

Usage:
    python src/predict.py --text "This product is absolutely amazing!"
    python src/predict.py --file data/new_reviews.csv
"""

import argparse
import joblib
import pandas as pd
from preprocessing import clean_text

MODEL_PATH = "models/svm_model.pkl"


def load_model():
    """Load saved model bundle (SVM + vectorizer + label encoder)."""
    bundle = joblib.load(MODEL_PATH)
    return bundle['model'], bundle['vectorizer'], bundle['label_encoder']


def predict_sentiment(text: str, model, vectorizer, label_encoder) -> dict:
    """
    Predict sentiment for a single text string.

    Returns:
        dict with 'sentiment', 'confidence', 'probabilities'
    """
    cleaned = clean_text(text)
    features = vectorizer.transform([cleaned])

    pred_label = model.predict(features)[0]
    pred_proba = model.predict_proba(features)[0]

    sentiment = label_encoder.inverse_transform([pred_label])[0]
    confidence = pred_proba.max()

    proba_dict = {
        label_encoder.inverse_transform([i])[0]: round(float(p), 4)
        for i, p in enumerate(pred_proba)
    }

    return {
        "text": text,
        "cleaned_text": cleaned,
        "sentiment": sentiment.capitalize(),
        "confidence": round(float(confidence), 4),
        "probabilities": proba_dict
    }


def predict_batch(filepath: str, model, vectorizer, label_encoder) -> pd.DataFrame:
    """Predict sentiment for all rows in a CSV file."""
    df = pd.read_csv(filepath)
    if 'review_text' not in df.columns:
        raise ValueError("CSV must contain a 'review_text' column.")

    results = []
    for _, row in df.iterrows():
        result = predict_sentiment(str(row['review_text']), model, vectorizer, label_encoder)
        results.append(result)

    output_df = pd.DataFrame(results)
    output_path = filepath.replace('.csv', '_predictions.csv')
    output_df.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")
    return output_df


def main():
    parser = argparse.ArgumentParser(description="Predict sentiment using trained SVM model")
    parser.add_argument('--text', type=str, help='Single review text to predict')
    parser.add_argument('--file', type=str, help='Path to CSV file for batch prediction')
    args = parser.parse_args()

    if not args.text and not args.file:
        parser.print_help()
        return

    print("Loading model...")
    model, vectorizer, label_encoder = load_model()
    print("Model loaded successfully.\n")

    if args.text:
        result = predict_sentiment(args.text, model, vectorizer, label_encoder)
        print(f"Input Text    : {result['text']}")
        print(f"Sentiment     : {result['sentiment']}")
        print(f"Confidence    : {result['confidence'] * 100:.1f}%")
        print(f"Probabilities : {result['probabilities']}")

    elif args.file:
        df = predict_batch(args.file, model, vectorizer, label_encoder)
        print(f"\nProcessed {len(df)} reviews.")
        print(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")


if __name__ == "__main__":
    main()
