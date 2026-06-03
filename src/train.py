"""
train.py
--------
Model training, evaluation, and comparison for sentiment analysis.
Trains SVM, Logistic Regression, and Naive Bayes classifiers.
Saves the best model (SVM) to models/svm_model.pkl
"""

import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import LabelEncoder

from preprocessing import load_and_preprocess, build_tfidf_features

# ── Config ──────────────────────────────────────────────────────────────────
DATA_PATH   = "data/reviews.csv"
MODEL_PATH  = "models/svm_model.pkl"
RESULTS_DIR = "results"
RANDOM_SEED = 42

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """Run full evaluation and return metrics dict."""
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"{'='*50}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

    return {"model": model_name, "accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1, "predictions": y_pred}


def plot_confusion_matrix(y_test, y_pred, labels, model_name: str):
    """Save confusion matrix heatmap."""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix — {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {path}")


def plot_model_comparison(results: list):
    """Save bar chart comparing accuracy of all models."""
    models  = [r["model"] for r in results]
    metrics = ["accuracy", "precision", "recall", "f1"]
    colors  = ['#2980B9', '#1D9E75', '#BA7517']

    x = np.arange(len(metrics))
    width = 0.22

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (res, color) in enumerate(zip(results, colors)):
        vals = [res[m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=res["model"], color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "model_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Model comparison chart saved to {path}")


def main():
    # ── 1. Load & preprocess ─────────────────────────────────────────────
    df = load_and_preprocess(DATA_PATH)

    le = LabelEncoder()
    df['label'] = le.fit_transform(df['sentiment'])
    label_names = le.classes_.tolist()

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df['cleaned_text'], df['label'],
        test_size=0.2, random_state=RANDOM_SEED, stratify=df['label']
    )

    # ── 2. TF-IDF features ───────────────────────────────────────────────
    X_train, X_test, vectorizer = build_tfidf_features(
        X_train_text, X_test_text, max_features=10000, ngram_range=(1, 2)
    )

    # ── 3. Train all models ───────────────────────────────────────────────
    results = []

    # — Naive Bayes —
    print("\nTraining Naive Bayes...")
    nb = MultinomialNB(alpha=0.1)
    nb.fit(X_train, y_train)
    results.append(evaluate_model(nb, X_test, y_test, "Naive Bayes"))

    # — Logistic Regression —
    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_SEED)
    lr.fit(X_train, y_train)
    results.append(evaluate_model(lr, X_test, y_test, "Logistic Regression"))

    # — SVM with GridSearchCV ─────────────────────────────────────────────
    print("\nTraining SVM with GridSearchCV hyperparameter tuning...")
    print("(This may take a few minutes...)")

    param_grid = {
        'C':      [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma':  ['scale', 'auto']
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    grid_search = GridSearchCV(
        SVC(probability=True, random_state=RANDOM_SEED),
        param_grid,
        cv=skf,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    best_svm = grid_search.best_estimator_
    print(f"\nBest SVM Parameters: {grid_search.best_params_}")
    print(f"Best CV F1-Score   : {grid_search.best_score_:.4f}")

    svm_result = evaluate_model(best_svm, X_test, y_test, "SVM (Best)")
    results.append(svm_result)

    # ── 4. Cross-validation on best SVM ──────────────────────────────────
    print("\nRunning 5-fold cross-validation on best SVM...")
    import scipy.sparse as sp
    X_all = sp.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])

    cv_scores = cross_val_score(best_svm, X_all, y_all, cv=5, scoring='accuracy')
    print(f"CV Accuracy Scores : {cv_scores}")
    print(f"Mean CV Accuracy   : {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

    # ── 5. Plots ──────────────────────────────────────────────────────────
    plot_confusion_matrix(y_test, svm_result["predictions"], label_names, "SVM")
    plot_model_comparison(results)

    # ── 6. Save model ─────────────────────────────────────────────────────
    joblib.dump({'model': best_svm, 'vectorizer': vectorizer, 'label_encoder': le}, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
