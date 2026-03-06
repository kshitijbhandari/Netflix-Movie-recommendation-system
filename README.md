# Netflix Movie Recommendation System

A machine learning project that predicts movie ratings for users based on the [Netflix Prize dataset](https://www.kaggle.com/netflix-inc/netflix-prize-data). Multiple collaborative filtering and matrix factorization models are built, compared, and stacked using XGBoost.

---

## Problem Statement

Netflix provided anonymous rating data and challenged participants to beat the accuracy of their in-house system, Cinematch, by 10%.

**Goal:** Predict the rating a user would give to a movie they haven't rated yet.
**Metric:** RMSE (Root Mean Square Error) and MAPE (Mean Absolute Percentage Error)

---

## Dataset

The dataset is the **Netflix Prize Data**, available on Kaggle:
[https://www.kaggle.com/netflix-inc/netflix-prize-data](https://www.kaggle.com/netflix-inc/netflix-prize-data)

**Files needed:**
| File | Description |
|------|-------------|
| `combined_data_1.txt` | Ratings batch 1 |
| `combined_data_2.txt` | Ratings batch 2 |
| `combined_data_3.txt` | Ratings batch 3 |
| `combined_data_4.txt` | Ratings batch 4 |
| `movie_titles.csv` | Movie ID to title mapping |

**Raw data format** (inside combined_data files):
```
<MovieID>:
<CustomerID>,<Rating>,<Date>
<CustomerID>,<Rating>,<Date>
...
```

**Scale:** ~100 million ratings, 480,000+ users, 17,770 movies. Ratings are integers from 1 to 5.

> The data files are large (~3 GB compressed). They are **not** included in this repository. Download from Kaggle and place them in the project root before running the notebooks.

---

## Project Structure

```
Netflix-Movie-recommendation-system/
├── Netflix_Movie.ipynb                  # Main notebook: full EDA + all ML models
├── netflix_movie_recommendation.ipynb   # SGD matrix factorization (bonus task)
└── README.md
```

### Netflix_Movie.ipynb

The main notebook covering the full ML pipeline:

1. **Business Problem** — problem framing, objectives, and constraints
2. **Data Overview** — data format, example data points
3. **Preprocessing** — merging all 4 data files into `(user_id, movie_id, rating, date)` format; deduplication; null checks
4. **Train/Test Split** — 80:20 split (chronological)
5. **Exploratory Data Analysis**
   - Rating distribution
   - Ratings per month and per day of week
   - User activity distribution (power-law / long-tail)
   - Movie popularity distribution
   - Sparse matrix construction and sparsity analysis
   - Global/user/movie average ratings
   - Cold-start analysis (new users and new movies in test set)
6. **Similarity Matrices**
   - User-User similarity (with TruncatedSVD for dimensionality reduction)
   - Movie-Movie similarity (cosine similarity on sparse matrix)
7. **Feature Engineering** — 13 features per (user, movie) pair:
   - `GAvg` — global average rating
   - `UAvg` — average rating given by the user
   - `MAvg` — average rating received by the movie
   - `sur1`–`sur5` — ratings given by the 5 most similar users to this movie
   - `smr1`–`smr5` — ratings given by the user to the 5 most similar movies
8. **ML Models** — see table below
9. **Model Comparison**

### netflix_movie_recommendation.ipynb

A focused notebook implementing **SGD-based matrix factorization** from scratch:

- Builds a user-movie adjacency matrix using `scipy.sparse`
- SVD decomposition via `sklearn`'s `randomized_svd`
- Manual SGD training loop to learn user/item biases (`b_i`, `c_j`) and latent vectors (`U`, `V`)
- Predicted rating formula: `y_hat = mu + b_i + c_j + u_i^T * v_j`
- Bonus: uses learned user matrix `U` to predict user gender via Logistic Regression (F1 score: 0.83–0.88)

---

## Models and Results

All models were trained on a **sample** of the data (10K users, 1K movies) to keep compute tractable. Results on test set:

| Model | RMSE |
|-------|------|
| SVD (Surprise) | **1.0726** |
| KNNBaseline — User-User (Surprise) | 1.0726 |
| KNNBaseline — Movie-Movie (Surprise) | 1.0728 |
| SVD++ (Surprise) | 1.0728 |
| BaselineOnly (Surprise) | 1.0730 |
| XGBoost + KNNBaseline (user+movie) | 1.0753 |
| XGBoost (Baseline + KNN + SVD + SVD++) | 1.0754 |
| XGBoost (13 features only) | 1.0762 |
| XGBoost + Baseline | 1.0763 |
| XGBoost + Baseline + KNN | 1.0764 |

> SVD-based matrix factorization gives the best single-model RMSE. Stacking with XGBoost does not meaningfully improve on pure Surprise models at this sample size.

---

## Setup

### Requirements

Install all dependencies with:

```bash
pip install -r requirements.txt
```

### Running the Notebooks

1. Download the Netflix Prize dataset from Kaggle and place the raw files in the project root.
2. Open `Netflix_Movie.ipynb` in Jupyter and run cells in order.
3. The preprocessing step (merging all 4 data files) will produce a single `ratings.csv`.
4. Subsequent cells use `ratings.csv` for all further processing.

> **Note:** The full notebook takes approximately **42 minutes** to run end-to-end on a standard machine. Sampled subsets (10K users, 1K movies) are used for model training to keep runtime reasonable.

---

## Key References

- [Netflix Prize Rules](https://www.netflixprize.com/rules.html)
- [Netflix Prize Dataset on Kaggle](https://www.kaggle.com/netflix-inc/netflix-prize-data)
- [Netflix Recommendations Blog (Medium)](https://medium.com/netflix-techblog/netflix-recommendations-beyond-the-5-stars-part-1-55838468f429)
- [Koren et al. — Matrix Factorization Techniques for Recommender Systems](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)
- [Surprise library documentation](https://surprise.readthedocs.io)
