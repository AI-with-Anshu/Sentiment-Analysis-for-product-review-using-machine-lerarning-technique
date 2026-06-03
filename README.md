# Sentiment Analysis for Product Reviews Using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-orange)](https://scikit-learn.org/)
[![IEEE Published](https://img.shields.io/badge/IEEE-Published-brightgreen)](https://ieeexplore.ieee.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **IEEE Xplore Published Research** | End-to-end ML pipeline for sentiment classification on Amazon product reviews achieving **89% accuracy** using SVM.

---

## Table of Contents

- [Overview](#overview)
- [Results](#results)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Model Comparison](#model-comparison)
- [Publication](#publication)
- [Author](#author)

---

## Overview

This project builds a complete machine learning pipeline to classify Amazon product reviews as **Positive**, **Negative**, or **Neutral** using classical NLP and supervised learning techniques.

**Key Highlights:**
- Processed and cleaned **10,000+ Amazon product reviews**
- Achieved **89% classification accuracy** using Support Vector Machine (SVM)
- Outperformed Logistic Regression and Naïve Bayes baselines by **7 percentage points**
- Applied **5-fold cross-validation** to ensure robust generalization
- Published findings in a **peer-reviewed IEEE Xplore research paper**

---

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **SVM (Best)** | **89%** | **0.88** | **0.89** | **0.88** |
| Logistic Regression | 82% | 0.81 | 0.82 | 0.81 |
| Naïve Bayes | 79% | 0.78 | 0.79 | 0.78 |

- **Overfitting gap reduced:** 11% → 4% after cross-validation + hyperparameter tuning
- **Vocabulary noise reduced:** ~35% after preprocessing pipeline

---

## Dataset

**Source:** [Amazon Product Reviews Dataset](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews)

- **Size:** 10,000+ reviews
- **Labels:** Positive, Negative, Neutral
- **Format:** CSV with `review_text` and `sentiment` columns

> Download the dataset from the link above and place `reviews.csv` inside the `data/` folder before running.

---

## Project Structure

```
sentiment-analysis-product-reviews/
│
├── data/
│   └── reviews.csv                  # Raw dataset (download separately)
│
├── notebooks/
│   └── sentiment_analysis.ipynb     # Full analysis notebook (run this)
│
├── src/
│   ├── preprocessing.py             # Text cleaning and feature engineering
│   ├── train.py                     # Model training and evaluation
│   └── predict.py                   # Predict sentiment on new text
│
├── models/
│   └── svm_model.pkl                # Saved trained model (generated after training)
│
├── results/
│   ├── confusion_matrix.png         # Model evaluation plots
│   └── model_comparison.png         # Bar chart comparing all 3 models
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/sentiment-analysis-product-reviews.git
cd sentiment-analysis-product-reviews
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download NLTK data
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"
```

---

## Usage

### Option A — Run the Jupyter Notebook (Recommended)
```bash
jupyter notebook notebooks/sentiment_analysis.ipynb
```
Run all cells top to bottom. The notebook covers:
- Data loading and EDA
- Text preprocessing
- Feature extraction (TF-IDF)
- Model training (SVM, LR, NB)
- Evaluation and comparison
- Saving the trained model

### Option B — Run Python scripts directly

**Train the model:**
```bash
python src/train.py
```

**Predict sentiment on new text:**
```bash
python src/predict.py --text "This product is absolutely amazing, highly recommend!"
# Output: Predicted Sentiment: Positive (Confidence: 0.94)
```

---

## Methodology

### 1. Text Preprocessing Pipeline
```
Raw Text
   ↓
Lowercase conversion
   ↓
Remove punctuation, numbers, special characters
   ↓
Tokenization
   ↓
Stop word removal (NLTK English stopwords)
   ↓
Lemmatization (WordNetLemmatizer)
   ↓
TF-IDF Vectorization (max_features=10000, ngram_range=(1,2))
   ↓
Cleaned Feature Matrix
```

### 2. Model Training
- **Algorithm:** Support Vector Machine with RBF kernel
- **Hyperparameter Tuning:** GridSearchCV over `C`, `gamma`, `kernel`
- **Validation:** 5-fold Stratified Cross-Validation
- **Best Parameters:** `C=10`, `gamma='scale'`, `kernel='rbf'`

### 3. Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score (weighted)
- Confusion Matrix
- ROC-AUC (One-vs-Rest for multiclass)

---

## Model Comparison

```
Accuracy (%)
90 |          ████
85 |          ████
80 |     ████ ████ ████
75 |     ████ ████ ████
   +-----+----+----+----
        NB   LR   SVM
        79%  82%  89%
```

SVM with TF-IDF features significantly outperforms simpler classifiers on this task due to its ability to handle high-dimensional sparse feature spaces effectively.

---

## Publication

This project was published as a peer-reviewed research paper in IEEE Xplore.

**Title:** Sentiment Analysis for Product Reviews Using Machine Learning Techniques  
**Publisher:** IEEE Xplore  
**Year:** 2025–2026  
**DOI:** [Add your DOI here]  
**Link:** [Add your IEEE Xplore link here]

---

## Technologies Used

| Technology | Purpose |
|------------|---------|
| Python 3.8+ | Core programming language |
| Pandas & NumPy | Data loading and manipulation |
| NLTK | Text preprocessing (tokenization, stopwords, lemmatization) |
| Scikit-Learn | ML models, TF-IDF, GridSearchCV, cross-validation |
| Matplotlib & Seaborn | Visualization and result plots |
| Joblib | Model serialization |

---

## Author

**Your Full Name**  
M.Tech Artificial Intelligence — Sage University, Indore  
📧 youremail@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/yourprofile)  
🐙 [GitHub](https://github.com/yourusername)

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
