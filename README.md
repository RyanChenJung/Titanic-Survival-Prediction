# 🚢 Titanic Survival Prediction: A Machine Learning Approach

## 📖 Project Overview

This project applies machine learning techniques to predict the survival of passengers on the Titanic. The goal was to build a robust classification model that generalizes well to unseen data, with a strong emphasis on **feature engineering** and **overfitting prevention**.

Using a **Random Forest Classifier**, the final model achieved a validation accuracy of **83.73%** with a minimized overfitting gap (~2.9%), outperforming baseline models like Logistic Regression and Decision Trees.

## 🔑 Key Features

* **Advanced Feature Engineering:** Extracted `Title` from names, grouped `Family_Size`, and engineered `Deck` information to capture hidden social-economic factors.
* **Rigorous Imputation:** Handled missing values in `Age` and `Fare` using group medians based on `Pclass` (Socio-economic status) rather than simple global averages.
* **Model Tournament:** Compared 7+ algorithms (LR, SVC, KNN, RF, etc.) using **10-Fold Stratified Cross-Validation**.
* **Overfitting Control:** Specifically addressed the "memorization" issue in Random Forest by tuning `max_depth`, `min_samples_leaf`, and `max_features` using GridSearchCV.

## 🛠️ Technologies Used

* **Python:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (RandomForest, GridSearchCV, StratifiedKFold)
* **Visualization:** Matplotlib, Seaborn
* **Serialization:** Joblib (Model persistence)

## 📊 Methodology

### 1. Data Preprocessing & Feature Engineering

* **Titles:** Extracted titles (Mr, Mrs, Miss, Master) to separate "Adult Males" from "Male Children", significantly improving prediction power.
* **Family Grouping:** Created `Family_Type` (Solo, Small, Big) to capture the survival advantage of small families.
* **Fare Log-Transformation:** Applied `np.log1p` to handle the skewed distribution of ticket fares.
* **Scaling:** Used `StandardScaler` for distance-based algorithms (SVM, KNN).

### 2. Model Selection (The Tournament)

We evaluated multiple models to establish a baseline.

| Model | Validation Accuracy | Overfitting Gap | Notes |
| --- | --- | --- | --- |
| **Logistic Regression** | 82.5% | 1.3% | Strong baseline, highly robust. |
| **Random Forest (Base)** | 80.2% | **14.4%** | **High Overfitting.** Needs tuning. |
| **SVC** | 82.4% | 2.5% | Good performance but slower training. |

### 3. Hyperparameter Tuning

We chose **Random Forest** for the final model due to its potential. To fix the 14% overfitting gap, we performed an extensive **GridSearchCV**:

* **Best Params:** `n_estimators=100`, `max_depth=10`, `min_samples_leaf=4`, `max_features='sqrt'`.
* **Result:** Validation Accuracy improved to **83.73%**, and the Overfitting Gap dropped to **2.9%**.

## 📈 Results & Insights

### Feature Importance

The model revealed that **Social Status** and **Gender/Age Group** were the primary drivers of survival:

1. **`Title_Mr` (Adult Men):** The most significant negative predictor.
2. **`Sex_male`:** Raw gender variable.
3. **`Fare`:** Proxy for wealth/status.
4. **`Pclass_3`:** Strong negative predictor for 3rd class passengers.

*(The model prioritized the engineered `Title_Mr` feature over the raw `Sex_male` feature, validating the feature engineering strategy.)*

## 📂 Project Structure

```
├── data/
│   ├── train.csv           # Training data
│   ├── holdout_test.csv    # Test data (unseen)
├── models/
│   ├── titanic_model_final.pkl  # Serialized Final Model
├── notebooks/
│   ├── Titanic_Analysis.ipynb   # Full analysis and code
├── results/
│   ├── Titanic_Results_from_Ryan_Chen.csv  # Final predictions
├── README.md
└── requirements.txt

```

## 🚀 How to Run

1. Clone the repo:
```bash
git clone https://github.com/YourUsername/Titanic-Survival-Prediction.git

```


2. Install dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn

```


3. Run the notebook or script to generate predictions.

## 👤 Author

**Ryan Chen**

* Master of Science in Applied Data Science, University of Chicago
---

*This project was completed as part of the MS in Applied Data Science curriculum.*
