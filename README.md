<div align="center">

# Machine Learning Projects

### Six end-to-end ML projects spanning NLP, recommendation systems, deep learning, and regression

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-154F5B?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

</div>

---

## Overview

| # | Project | Problem Type | Approach | Result |
|---|---------|--------------|----------|--------|
| 1 | [Image Classification](#1--image-classification--handwritten-digit-recognition) | Multi-class classification | CNN (TensorFlow/Keras) | **98.83%** test accuracy |
| 2 | [Sentiment Analysis](#2--sentiment-analysis--tweet-classification) | NLP / text classification | TF-IDF + Logistic Regression | **71%** accuracy, 0.70 weighted F1 |
| 3 | [Music Recommendation](#3--music-recommendation-system) | Recommendation | Content-based, cosine similarity | Top-K similar songs |

### 🎓 Internship Work — [Oasis Infobyte (OIBSIP)](./oasis_infobyte_internship/)

| # | Project | Problem Type | Model | Result |
|---|---------|--------------|-------|--------|
| 4 | [Sales Prediction](./oasis_infobyte_internship/sales_prediction/) | Regression | Linear Regression | **R² = 0.91** |
| 5 | [Car Price Prediction](./oasis_infobyte_internship/car_price_prediction/) | Regression | Random Forest Regressor | MAE / MSE / R² evaluated |
| 6 | [Spam Email Detection](./oasis_infobyte_internship/spam_email_detection/) | Text classification | TF-IDF + Logistic Regression | **96.32%** test accuracy |

---

## 1 · Image Classification — Handwritten Digit Recognition

**Problem:** Automatically recognize handwritten digits (0–9) from images — the foundational task behind postal code sorting and cheque digitization.

**Dataset:** MNIST — 60,000 training + 10,000 test grayscale images (28×28 px), loaded directly via `keras.datasets`.

### Model Architecture

```
Input (28×28×1)
  ↓ Conv2D(32, 3×3, ReLU)  →  MaxPooling2D(2×2)
  ↓ Conv2D(64, 3×3, ReLU)  →  MaxPooling2D(2×2)
  ↓ Conv2D(64, 3×3, ReLU)
  ↓ Flatten  →  Dense(64, ReLU)  →  Dense(10, Softmax)
```

**Optimizer:** Adam · **Loss:** Sparse categorical cross-entropy · **Epochs:** 5

### Results

| Epoch | Train Accuracy | Train Loss |
|-------|----------------|------------|
| 1 | 88.98% | 0.3449 |
| 2 | 98.46% | 0.0504 |
| 3 | 99.01% | 0.0332 |
| 4 | 99.21% | 0.0253 |
| 5 | 99.43% | 0.0185 |

> **Test accuracy: 98.83%** (loss 0.0491) — the small train/test gap indicates the model generalized well without significant overfitting.

**Key steps:** pixel normalization to `[0,1]` · channel-dimension reshaping · prediction visualization with Matplotlib · model exported to `mnist_model.h5`

📓 [`image_classification_project/imageclassification.ipynb`](./image_classification_project/imageclassification.ipynb)

---

## 2 · Sentiment Analysis — Tweet Classification

**Problem:** Classify social media posts as positive, negative, or neutral — the core of brand monitoring and customer feedback analysis.

**Dataset:** 500 labelled tweets with text, sentiment, timestamp, and platform metadata.

### Pipeline

```
Raw text
  ↓ Lowercase  →  strip non-alphabetic characters
  ↓ Stopword removal (NLTK)  →  WordNet lemmatization
  ↓ TF-IDF vectorization (unigrams + bigrams, 5,000 features)
  ↓ Logistic Regression (class_weight='balanced', max_iter=1000)
```

Stratified 80/20 train-test split preserves class proportions.

### Results

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Negative | 0.85 | 0.41 | 0.55 | 27 |
| Neutral | 0.63 | 0.90 | 0.74 | 40 |
| Positive | 0.80 | 0.73 | 0.76 | 33 |
| **Accuracy** | | | **0.71** | 100 |
| **Weighted avg** | 0.75 | 0.71 | 0.70 | 100 |

> **Honest read:** the model is precise on negative tweets (0.85) but recalls only 41% of them — it misses many, usually labelling them neutral. With just 27 negative samples in the test set, more data is the clearest path to improvement. `class_weight='balanced'` was used specifically to counteract this imbalance.

**Includes** a `predict_sentiment()` helper for inference on unseen text.

📓 [`sentiment_Analysis_project/notebooks/sentiment_analysis.ipynb`](./sentiment_Analysis_project/notebooks/sentiment_analysis.ipynb)

---

## 3 · Music Recommendation System

**Problem:** Given a song a listener enjoys, suggest similar tracks — the content-based approach used when you lack the user-interaction history that collaborative filtering requires.

**Dataset:** 100 songs with title, artist, genre, release year, duration, listen date, and platform.

### Approach

```
Raw listening data
  ↓ Deduplicate to unique songs
  ↓ Feature engineering — aggregate listen counts per song
  ↓ One-hot encode categorical features (genre, artist)
  ↓ StandardScaler normalization
  ↓ Cosine similarity matrix across all song pairs
  ↓ recommend_song(title, top_k) → K nearest neighbours
```

**Why content-based:** it sidesteps the cold-start problem — a new song can be recommended immediately from its own attributes, with no listening history needed.

📓 [`music_recommendation_system/notebooks/music_recommendation.ipynb`](./music_recommendation_system/notebooks/music_recommendation.ipynb)

---

## Repository Structure

```
Machine-learning-project/
├── image_classification_project/
│   └── imageclassification.ipynb
├── sentiment_Analysis_project/
│   ├── dataset/sentiment_analysis.csv
│   └── notebooks/sentiment_analysis.ipynb
├── music_recommendation_system/
│   ├── dataset/music_dataset.csv
│   └── notebooks/music_recommendation.ipynb
└── oasis_infobyte_internship/          # OIBSIP Data Science Internship
    ├── sales_prediction/
    ├── car_price_prediction/
    └── spam_email_detection/
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab

### Installation

```bash
# Clone the repository
git clone https://github.com/ansht120/Machine-learning-project.git
cd Machine-learning-project

# Install dependencies
pip install numpy pandas matplotlib scikit-learn tensorflow nltk jupyter

# Launch Jupyter
jupyter notebook
```

### Running Each Project

**Image Classification** — no setup needed; MNIST downloads automatically on first run.

```bash
jupyter notebook image_classification_project/imageclassification.ipynb
```

**Sentiment Analysis** — downloads NLTK corpora on first run.

```bash
cd sentiment_Analysis_project/notebooks
jupyter notebook sentiment_analysis.ipynb
```

> The notebook loads `sentiment_analysis.csv`. Either run it from the `dataset/` folder or update the path to `../dataset/sentiment_analysis.csv`.

**Music Recommendation**

```bash
cd music_recommendation_system/notebooks
jupyter notebook music_recommendation.ipynb
```

> The notebook loads `MusicDataset.csv`. Update the path to `../dataset/music_dataset.csv` to match this repository's layout.

---

## Skills Demonstrated

| Area | Techniques |
|------|-----------|
| **Deep Learning** | CNN architecture design, convolution/pooling layers, softmax classification, Adam optimization |
| **NLP** | Text cleaning, stopword removal, lemmatization, TF-IDF with n-grams |
| **Classical ML** | Logistic regression, class imbalance handling, stratified splitting |
| **Recommender Systems** | Content-based filtering, one-hot encoding, cosine similarity, cold-start reasoning |
| **Evaluation** | Accuracy, precision, recall, F1-score, per-class analysis, train/test gap interpretation |
| **Data Handling** | Pandas transformations, feature scaling, deduplication, aggregation |

---

## Future Improvements

- **Sentiment Analysis** — expand beyond 500 samples; try SVM and transformer-based models (BERT) as baselines to beat
- **Image Classification** — add data augmentation and dropout; extend to Fashion-MNIST or CIFAR-10
- **Music Recommendation** — introduce collaborative filtering as a hybrid layer once interaction data is available
- **Across all** — refactor notebooks into reusable modules, add `requirements.txt`, and pin dependency versions

---

## Author

**Ansh Thakur** — Aspiring Data Analyst & ML Enthusiast

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/ansht120)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ansh-thakur-5407)
