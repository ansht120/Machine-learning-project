# Sales Prediction

> Regression model that forecasts product sales from outlet and item attributes.

![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=flat-square)

**Result: R² = 0.91 · MSE = 2.86**

*Completed as Task 1 of the Oasis Infobyte Data Science Internship (OIBSIP).*

---

## Problem

Sales forecasting drives inventory planning, and both directions of error are expensive:

- **Under-forecast** → stockouts, lost revenue, and customers who go elsewhere
- **Over-forecast** → capital tied up in unsold inventory, plus storage and spoilage costs

The task is to predict sales from item and outlet characteristics: MRP, outlet size, item type, and outlet location.

---

## Approach

```
Raw sales data
  ↓ Load and inspect
  ↓ Clean — handle missing values and duplicates
  ↓ Exploratory data analysis — distributions, correlations
  ↓ Encode categorical variables (outlet type, item category, location)
  ↓ Train Linear Regression
  ↓ Evaluate — MSE, MAE, RMSE, R²
```

### Why Linear Regression suffices here

The obvious question is whether a more powerful model would do better. The results argue no: **an R² of 0.91 from a linear model tells you the underlying relationship is already largely linear.**

MRP relates to sales in a roughly proportional way, and the outlet attributes shift that relationship up or down without introducing complex interactions. A Random Forest could capture non-linearities — but where few exist, it adds computational cost and loses interpretability for marginal gain.

This is worth stating explicitly because it contrasts with the [car price prediction](../car_price_prediction_project/) project in this repository, where depreciation *is* strongly non-linear and a Random Forest is genuinely the better choice. **Matching model complexity to the structure of the problem is the actual skill** — not reaching for the most powerful algorithm by default.

---

## Results

| Metric | Value |
|--------|-------|
| **R² Score** | **0.91** |
| Mean Squared Error | 2.86 |

### Interpretation

**R² of 0.91 means the model explains 91% of the variance in sales.** The remaining 9% comes from factors not present in the dataset — seasonality, local promotions, competitor activity, weather.

For retail forecasting this is a strong result. It suggests the features captured here (price, outlet characteristics, product category) account for most of what drives sales volume, and that the linear relationship assumption holds well.

---

## Running It

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
jupyter notebook sales_prediction.ipynb
```

> **Dataset not included.** The notebook expects `sales dataset.csv` in the same folder. Supply your own copy or update the `read_csv()` path in the first cell. Saved notebook outputs preserve the results above.

---

## Next Steps

- **Residual analysis** — plot residuals against predictions to confirm the linearity assumption holds across the full range rather than just on average
- **Cross-validation** — a single train/test split can be optimistic; k-fold would give a more reliable R² estimate
- **Feature importance** — inspect coefficients to identify which outlet attributes matter most for planning decisions
- **Add temporal features** — seasonality and promotional calendars likely explain a meaningful share of the residual 9%

---

[⬅ Back to all ML projects](../README.md)
