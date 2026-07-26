# Spam Email Detection

> Binary text classifier that separates spam from legitimate email using TF-IDF features and logistic regression.

![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-154F5B?style=flat-square&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)

**Result: 96.32% test accuracy**

*Completed as Task 3 of the Oasis Infobyte Data Science Internship (OIBSIP).*

---

## Problem

Spam filtering is the textbook example of an **asymmetric-cost classification problem**. The two error types are not equally bad:

- **False negative** (spam reaches the inbox) — mildly annoying, the user deletes it
- **False positive** (legitimate mail marked spam) — potentially serious; a missed job offer or invoice sits unseen in a spam folder

That asymmetry should shape how the model is tuned: precision on the "spam" class matters more than recall, because being wrong about legitimate mail is the costlier mistake.

---

## Approach

```
Raw email text
  ↓ Data cleaning — remove nulls and duplicates
  ↓ Text preprocessing (NLTK)
  ↓ TF-IDF vectorisation
  ↓ Logistic Regression classifier
  ↓ Evaluate on held-out test set
```

### Why TF-IDF over raw word counts

**TF-IDF** (Term Frequency–Inverse Document Frequency) weights each word by how distinctive it is:

- Words appearing in *every* email ("the", "and", "to") get near-zero weight — no discriminative signal
- Words concentrated in one class ("winner", "unsubscribe", "claim") get high weight

A raw `CountVectorizer` would treat "the" appearing 40 times as a strong feature. TF-IDF correctly recognises it as noise. This is why TF-IDF is the standard baseline for text classification.

### Why Logistic Regression

For high-dimensional sparse text features, logistic regression trains fast, resists overfitting with proper regularisation, and — importantly — is **interpretable**. The learned coefficients show exactly which words push a message toward "spam", which matters when a user asks why their email was filtered.

---

## Results

| Split | Accuracy |
|-------|----------|
| Training | 96.11% |
| **Test** | **96.32%** |

### Interpretation

**Test accuracy slightly exceeds training accuracy.** That's a healthy signal — the model has *not* memorised the training set. Overfitting shows the opposite pattern: high training accuracy with a marked drop on unseen data.

Near-identical scores across both splits indicate the model found genuinely generalisable patterns in spam vocabulary rather than artefacts of the training sample.

> **Caveat worth stating:** accuracy alone is incomplete here. Given the asymmetric costs above, a full evaluation would report the confusion matrix and per-class precision/recall — specifically how many legitimate emails were wrongly flagged. That number determines whether the filter is safe to deploy.

---

## Running It

```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn
jupyter notebook spam_email_detection.ipynb
```

> **Dataset not included.** The notebook expects `spam_mail.csv` (read with `ISO-8859-1` encoding) in the same folder. Supply your own copy or update the `read_csv()` path in the first cell. Saved notebook outputs preserve the results above.

---

## Next Steps

- **Report the confusion matrix** — given asymmetric costs, the false-positive count is the metric that actually matters
- **Threshold tuning** — raise the spam decision threshold to trade recall for precision, protecting legitimate mail
- **Benchmark Multinomial Naive Bayes** — the classic spam-detection baseline, worth comparing against this implementation
- **Feature inspection** — surface the highest-weighted spam indicators to sanity-check what the model learned

---

[⬅ Back to all ML projects](../README.md)
