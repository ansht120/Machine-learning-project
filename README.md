<div align="center">

# Oasis Infobyte — Data Science Internship

### OIBSIP · Machine Learning Task Submissions

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-154F5B?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

</div>

---

## About This Internship

This repository contains my task submissions for the **Oasis Infobyte Data Science Internship (OIBSIP)** — a remote programme in which interns build and document end-to-end machine learning solutions independently.

Each task covers the full supervised-learning workflow: data cleaning, exploratory analysis, feature engineering, model selection, evaluation against appropriate metrics, and interpretation of results. The three tasks were deliberately varied — two regression problems and one text-classification problem — to cover different problem types and evaluation frameworks.

---

## Tasks at a Glance

| # | Task | Problem Type | Model | Key Metric |
|---|------|--------------|-------|-----------|
| 1 | [Sales Prediction](#task-1--sales-prediction) | Regression | Linear Regression | **R² = 0.91** · MSE 2.86 |
| 2 | [Car Price Prediction](#task-2--car-price-prediction) | Regression | Random Forest Regressor | MAE · MSE · R² |
| 3 | [Spam Email Detection](#task-3--spam-email-detection) | Binary text classification | TF-IDF + Logistic Regression | **96.32%** test accuracy |

---

## Task 1 · Sales Prediction

**Objective:** Predict product sales from outlet and item attributes — MRP, outlet size, item type, and outlet location.

**Why this matters:** Sales forecasting drives inventory planning. Under-forecast and you stock out; over-forecast and you tie up capital in unsold goods.

### Approach

```
Raw sales data
  ↓ Handle missing values and duplicates
  ↓ Exploratory data analysis (distribution, correlation)
  ↓ Encode categorical variables (outlet type, item category)
  ↓ Train Linear Regression
  ↓ Evaluate — MSE, MAE, RMSE, R²
```

### Results

| Metric | Value |
|--------|-------|
| **R² Score** | **0.91** |
| Mean Squared Error | 2.86 |

> An R² of 0.91 means the model explains 91% of the variance in sales. For a linear model on retail data this is strong — it suggests the relationship between MRP, outlet characteristics and sales is largely linear, so the added complexity of a tree-based model buys little here.

📓 [`SalesPredict_Task1.ipynb`](./SalesPredict_Task1.ipynb) · [Task README](./READMETask1.md)

---

## Task 2 · Car Price Prediction

**Objective:** Estimate a used car's selling price from year, mileage, fuel type, transmission, and engine specifications.

**Why this matters:** Resale pricing is the core of any used-vehicle marketplace — mispricing costs either the seller margin or the buyer trust.

### Approach

```
Vehicle listings data
  ↓ Clean and inspect
  ↓ Feature engineering — derive vehicle age from year
  ↓ Encode categoricals (fuel type, transmission, seller type)
  ↓ Train Random Forest Regressor
  ↓ Evaluate — MAE, MSE, R²
```

**Why Random Forest here:** Depreciation is not linear. A car loses value fastest in its first years and the curve flattens later; mileage interacts with age rather than acting independently. A tree ensemble captures those non-linear interactions without requiring them to be specified in advance — which is exactly why it suits this problem better than the linear model used in Task 1.

📓 [`Car_price _Prediction_Task2.ipynb`](./Car_price%20_Prediction_Task2.ipynb) · [Task README](./READMETask2.md)

---

## Task 3 · Spam Email Detection

**Objective:** Classify emails as spam or legitimate using natural language processing.

**Why this matters:** Spam filtering is the classic asymmetric-cost classification problem — a spam email reaching the inbox is a mild annoyance, but a legitimate email wrongly filtered can be a missed job offer.

### Approach

```
Raw email text
  ↓ Text preprocessing (NLTK) — cleaning, tokenization
  ↓ TF-IDF vectorization — weight terms by discriminative power
  ↓ Logistic Regression classifier
  ↓ Evaluate on held-out test set
```

### Results

| Split | Accuracy |
|-------|----------|
| Training | 96.11% |
| **Test** | **96.32%** |

> Test accuracy slightly *exceeding* training accuracy is a healthy sign — it indicates the model has not memorized the training set and generalizes to unseen emails. A large gap in the other direction would signal overfitting.

**Why TF-IDF over raw word counts:** TF-IDF down-weights words appearing across all emails ("the", "and") and up-weights terms that distinguish spam from legitimate mail — so the classifier learns from signal rather than noise.

📓 [`Spam_Email _Detection_Task3.ipynb`](./Spam_Email%20_Detection_Task3.ipynb) · [Task README](./READMETask3.md)

---

## Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3 |
| **Data Handling** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn (Linear Regression, Random Forest, Logistic Regression) |
| **NLP** | NLTK, TF-IDF vectorization |
| **Visualization** | Matplotlib, Seaborn |
| **Environment** | Jupyter Notebook |

---

## Skills Demonstrated

- **Model selection reasoning** — matching algorithm to problem structure (linear for linear relationships, ensembles for non-linear interactions)
- **Evaluation literacy** — choosing R²/MSE for regression and accuracy for balanced classification, and interpreting the train/test gap
- **Feature engineering** — categorical encoding, derived features, text vectorization
- **End-to-end workflow** — raw data through cleaning, EDA, modelling, and evaluation
- **Documentation** — each task written up with objective, method, and outcome

---

## Getting Started

```bash
# Clone the repository
git clone https://github.com/ansht120/oasis-infobyte-data-science-internship.git
cd oasis-infobyte-data-science-internship

# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn nltk jupyter

# Launch Jupyter
jupyter notebook
```

Open any `.ipynb` file and run all cells.

---

## Related Work

These projects are also mirrored in my main [Machine Learning portfolio](https://github.com/ansht120/Machine-learning-project), alongside CNN image classification, sentiment analysis, and a content-based recommendation system.

---

## Author

**Ansh Thakur** — Aspiring Data Analyst & ML Enthusiast

[![Portfolio](https://img.shields.io/badge/Portfolio-00C7B7?style=flat&logo=netlify&logoColor=white)](https://ansh-thakur.netlify.app/)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ansh-thakur-5407192b4/)
[![Email](https://img.shields.io/badge/Email-D14836?style=flat&logo=gmail&logoColor=white)](mailto:ansht1194@gmail.com)
