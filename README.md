# 🍽️ Predictive Analysis of Recipe Ratings
### What makes a popular meal?

> A machine learning project exploring the factors behind recipe ratings on food.com - built as part of the **Data Literacy** course at the **University of Tübingen (2022/23)**.

---

## 👥 Team

| Name |  
|---|
| Markus Heimerl |
| Olha Yarikova |
| Reshma Manjunatha Rao |

---

## 📌 Project Overview

Can we predict how well a recipe will be rated, before anyone tastes it? This project investigates which features of a recipe (nutrition, cook time, complexity) are the strongest predictors of its average user rating on food.com.

We built a full ML pipeline: from raw data ingestion and preprocessing, through feature engineering and regression analysis, to neural architecture search via Hyperband.

**Key finding:** Total cook time, sodium content, and sugar content are the strongest predictors of a recipe's average rating.

---

## 🗂️ Dataset

- **Source:** [food.com on Kaggle](https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews) - 500,000+ recipes, 1.4M+ reviews
- Three datasets merged: `recipes.csv`, `reviews.csv`, `recipes_w_search_terms`
- **Challenge:** Highly imbalanced - 72% of all reviews are 5-star ratings

---

## 🔧 Pipeline

### 1. Data Preprocessing
- Merged three datasets on `recipeId`
- Decoded cook time from string format (e.g. `2H30M`) to minutes
- Label-encoded `RecipeCategory`
- Excluded recipes with fewer than 5 reviews
- Applied 1%–99% quantile clipping to remove outliers
- Excluded zero-rated reviews (unrated text comments)

### 2. Feature Engineering
- `steps_word_length` - proxy for recipe complexity
- `ingredients_amount` - number of ingredients (effort + complexity signal)

### 3. Balancing
- Oversampled the minority class (low-rated recipes) to 5,000 records each above/below rating 3
- Ensured meaningful regression on a skewed target

### 4. Modeling

#### Linear Regression
- Fitted on **all combinations** of input features to rank predictors
- Evaluated via MSE, R² score, and permutation test distance
- Best R²: **0.11** | Minimum MSE: **0.9**

#### Neural Network (Hyperband Search)
- Search space: number of layers, nodes per layer
- Best architecture: 464-node input → 9 dense layers (8–28 nodes each)
- **Validation loss: 0.11 | R²: 0.9** | 22,425 trainable parameters
- Key insight: deeper stacks of dense layers were critical for performance

---

## 📊 Results

| Feature | Predictive Quality |
|---|---|
| ⏱️ TotalTime | ⭐ Best |
| 🧂 SodiumContent | ⭐ Best |
| 🍬 SugarContent | ✅ Strong |
| 🌾 FiberContent | ✅ Strong |
| 🥘 ingredients_amount | ✅ Strong |
| 🫙 FatContent | Moderate |
| 🍞 CarbohydrateContent | Moderate |
| 🔥 Calories | Moderate |
| 🧬 ProteinContent | Moderate |
| 🐓 CholesterolContent | Weak |
| 📝 steps_word_length | Weak |
| 🏷️ RecipeCategory | Weak |

> Features ranked by MSE contribution, R² contribution, and permutation test - all three metrics agreed, confirming robustness.
<p align="center">
<img width="600" height="500" alt="results-charts" src="https://github.com/user-attachments/assets/b383d9d7-1ed2-4d57-899b-aff32647dc27" />
</p>

---

## ⚠️ Limitations

- **Platform bias:** food.com is English-language only; findings don't generalize globally
- **Missing factors:** author reputation, instruction clarity, ingredient availability, and price are untracked
- **Sparse recent data:** very little data from 2020 onwards (COVID-era cooking behaviour unrepresented)
- **Unrated reviews:** 5.4% of reviews had no star rating; sentiment analysis (NLTK) showed 79.5% of these were positive - excluded to avoid further skewing

<p align="center">
<img width="600" height="500" alt="rating-distribution" src="https://github.com/user-attachments/assets/f1b20f20-ce1f-4fd0-895b-a26b995fa76e" />
</p>

---

## 🔮 Future Work

- Expand features: cuisine type, dietary restrictions, ingredient popularity
- Cross-platform data: incorporate ratings from other recipe sites or social media
- Reframe as **classification** (well-received vs. not) for potentially better model fit
- Use advanced imbalanced regression techniques (e.g. Balanced MSE)

---

## 🛠️ Tech Stack

`Python` · `pandas` · `scikit-learn` · `Keras / TensorFlow` · `NLTK` · `Hyperband` · `NumPy` · `Matplotlib`

---

## 📚 References

1. Kaggle: [food.com Reviews](https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews?select=reviews.csv)
2. Kaggle: [food.com Recipes](https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews?select=recipes.csv)
3. Kaggle: [food.com Search Terms & Tags](https://www.kaggle.com/datasets/shuyangli94/foodcom-recipes-with-search-terms-and-tags)
4. Li et al. (2016) - Hyperband: A novel bandit-based approach to hyperparameter optimization
5. Ren et al. (2022) - Balanced MSE for Imbalanced Visual Regression
6. Yang et al. (2021) - Delving into Deep Imbalanced Regression
7. Bird, Klein & Loper (2009) - Natural Language Processing with Python (NLTK)
8. Bonta et al. (2019) - Lexicon-based approaches for sentiment analysis
