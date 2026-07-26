# Music Recommendation System

> Content-based recommender that suggests similar songs using cosine similarity over engineered audio and metadata features.

![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)

---

## Problem

Given a song someone already enjoys, recommend others they're likely to enjoy. This is the core loop behind every streaming platform's discovery feature.

## Dataset

100 songs with title, artist, genre, release year, duration, listen date, and platform. Included at [`dataset/music_dataset.csv`](./dataset/music_dataset.csv).

---

## Approach

```
Raw listening data
  ↓ Deduplicate to unique songs
  ↓ Feature engineering — aggregate listen count per song
  ↓ One-hot encode categoricals (genre, artist)
  ↓ StandardScaler normalisation
  ↓ Cosine similarity matrix (all song pairs)
  ↓ recommend_song(title, top_k) → K most similar tracks
```

### Why content-based rather than collaborative filtering

The two dominant approaches to recommendation are:

| Approach | Uses | Weakness |
|----------|------|----------|
| **Collaborative filtering** | Patterns across many users' listening histories | **Cold start** — a brand-new song has no listening history, so it can never be recommended |
| **Content-based** *(used here)* | The song's own attributes | Tends toward similar recommendations; can't surface unexpected cross-genre finds |

Content-based was the right fit here because the dataset describes song *attributes* rather than multiple users' behaviour. It also means a newly added song is immediately recommendable from its metadata alone — no waiting for play counts to accumulate.

### Why StandardScaler matters

Cosine similarity is sensitive to feature magnitude. Without scaling, `Release_Year` (values around 2024) would numerically dominate `Duration_Minutes` (values around 4), making the recommender effectively sort by release date. Standardising puts every feature on a comparable scale so genre, artist, duration, and year each contribute meaningfully.

### Why cosine similarity rather than Euclidean distance

Cosine measures the *angle* between feature vectors, not their magnitude — so two songs with a similar profile of characteristics score as similar even if one has a much higher listen count. For sparse one-hot encoded data this behaves considerably better than raw distance.

---

## Usage

```python
recommend_song("Faded", top_k=3)
```

Returns the three most similar tracks by cosine similarity, excluding the query song itself.

---

## Running It

```bash
pip install pandas numpy scikit-learn
cd notebooks
jupyter notebook music_recommendation.ipynb
```

The notebook reads the dataset from `../dataset/music_dataset.csv`.

---

## Limitations & Next Steps

- **Small catalogue** — 100 songs limits how meaningful "most similar" can be. Recommendation quality scales sharply with catalogue size.
- **No evaluation metric** — content-based recommenders are hard to score without user feedback. Adding precision@k against held-out listening data would make quality measurable rather than anecdotal.
- **Hybrid approach** — once real interaction data exists, layering collaborative filtering on top would surface unexpected recommendations that pure content matching cannot.
- **Audio features** — incorporating tempo, key, and energy (e.g. via the Spotify API) would capture musical similarity that genre labels miss.

---

[⬅ Back to all ML projects](../README.md)
