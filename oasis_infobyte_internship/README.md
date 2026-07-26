# Oasis Infobyte — Data Science Internship (OIBSIP)

Three supervised-learning tasks completed during the Oasis Infobyte Data Science Internship, covering regression and text classification.

---

## Projects

| # | Task | Problem Type | Model | Result |
|---|------|--------------|-------|--------|
| 1 | [Sales Prediction](./sales_prediction/) | Regression | Linear Regression | **R² = 0.91**, MSE 2.86 |
| 2 | [Car Price Prediction](./car_price_prediction/) | Regression | Random Forest Regressor | Evaluated on MAE, MSE, R² |
| 3 | [Spam Email Detection](./spam_email_detection/) | Binary text classification | TF-IDF + Logistic Regression | **96.32%** test accuracy |

---

## 1 · Sales Prediction

Predicts product sales from outlet and item attributes — MRP, outlet size, item type, and outlet location.

**Approach:** Linear Regression with feature encoding and exploratory analysis.

**Result:** R² of 0.91 — the model explains 91% of the variance in sales, with a mean squared error of 2.86.

📓 [`sales_prediction/`](./sales_prediction/)

---

## 2 · Car Price Prediction

Estimates a used car's selling price from year, mileage, fuel type, transmission, and engine details.

**Approach:** Random Forest Regressor, chosen over linear models to capture non-linear relationships between age, mileage, and price. Evaluated with MAE, MSE, and R².

📓 [`car_price_prediction/`](./car_price_prediction/)

---

## 3 · Spam Email Detection

Classifies emails as spam or legitimate using natural language processing.

**Approach:** TF-IDF vectorization feeding a Logistic Regression classifier, with NLTK-based text preprocessing.

**Result:**

| Split | Accuracy |
|-------|----------|
| Training | 96.11% |
| **Test** | **96.32%** |

> Test accuracy slightly exceeding training accuracy indicates no overfitting — the model generalizes well to unseen emails.

📓 [`spam_email_detection/`](./spam_email_detection/)

---

## Tech Stack

`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `NLTK` · `Matplotlib` · `Seaborn`

---

[⬅ Back to all ML projects](../README.md)
