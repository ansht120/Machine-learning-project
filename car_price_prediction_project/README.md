# Car Price Prediction

> Random Forest regressor that estimates a used car's selling price from its specifications and history.

![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=flat-square)

*Completed as Task 2 of the Oasis Infobyte Data Science Internship (OIBSIP).*

---

## Problem

Resale pricing is the core mechanic of any used-vehicle marketplace, and mispricing costs someone either way:

- **Priced too high** → the listing sits unsold, and the seller carries holding costs
- **Priced too low** → the seller loses margin they can't recover

The task is to predict selling price from year, mileage, fuel type, transmission, and engine details.

---

## Approach

```
Vehicle listings data
  ↓ Import and inspect
  ↓ Clean — handle missing values
  ↓ Exploratory data analysis and visualisation
  ↓ Feature engineering — derive vehicle age from year
  ↓ Encode categoricals (fuel type, transmission, seller type)
  ↓ Feature selection
  ↓ Train Random Forest Regressor
  ↓ Evaluate — MAE, MSE, RMSE, R²
```

### Why Random Forest is the right choice here

**Depreciation is not linear.** Three properties of the domain make a tree ensemble the natural fit:

1. **The depreciation curve is steep then flat.** A car loses a large share of its value in the first two or three years, after which the decline slows considerably. A linear model forced through that curve will overestimate the value of new cars and underestimate old ones.

2. **Features interact.** High mileage on a two-year-old car signals something different from high mileage on a ten-year-old one. Linear regression treats age and mileage as independent additive contributions; a tree can branch on age first and then apply a different mileage rule within each branch.

3. **Categorical effects are conditional.** A diesel engine may add value in one segment and subtract it in another. Trees capture this natively; a linear model needs every interaction specified by hand.

Random Forest handles all three without requiring the relationships to be known in advance — and averaging across many trees controls the overfitting a single deep tree would suffer.

> **Contrast worth noting:** the [sales prediction](../sales_prediction_project/) project in this repository uses plain Linear Regression and achieves R² = 0.91, because *that* relationship genuinely is linear. Choosing model complexity to match problem structure — rather than defaulting to the most powerful algorithm — is the point.

---

## Evaluation

The model is evaluated on four complementary regression metrics:

| Metric | What it tells you |
|--------|-------------------|
| **MAE** | Average error in currency units — the most directly interpretable ("off by ~₹X on average") |
| **MSE** | Penalises large errors disproportionately — surfaces occasional bad misses |
| **RMSE** | Back in the original units, but still weighted toward large errors |
| **R²** | Share of price variance the model explains |

MAE and RMSE together are informative: if RMSE is much larger than MAE, a few predictions are badly wrong even though typical accuracy looks acceptable.

---

## Running It

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
jupyter notebook car_price_prediction.ipynb
```

> **Dataset not included.** The notebook expects `car_data.csv` in the same folder. Supply your own copy or update the `read_csv()` path in the first cell.

---

## Next Steps

- **Feature importance plot** — Random Forest exposes this directly; showing which factors drive price most would make the model's reasoning legible to a non-technical audience
- **Hyperparameter tuning** — grid search over `n_estimators` and `max_depth` for a measurable gain
- **Benchmark against Gradient Boosting** — typically outperforms Random Forest on tabular regression
- **Residual analysis by price band** — check whether the model is systematically worse on luxury or budget vehicles

---

[⬅ Back to all ML projects](../README.md)
