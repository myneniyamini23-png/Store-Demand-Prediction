# model_training.py
# --------------------------------------------------------
# This script:
# - Loads data
# - Applies feature engineering
# - Splits train/test by time
# - Trains baseline + base LightGBM
# - Performs random search hyperparameter tuning
# - Trains tuned LightGBM
# - Saves feature importance plot
# --------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from lightgbm import LGBMRegressor
import random

from feature_engineering import add_time_features, add_lag_features


# ------------ Load Data ------------
def load_dataset(path="data/train.csv"):
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ------------ Feature Engineering ------------
def preprocess(df):
    df = add_time_features(df)
    df = add_lag_features(df)
    df = df.dropna().reset_index(drop=True)
    return df


# ------------ Baseline Model ------------
def baseline_rmse(train_df, test_df):
    item_mean = train_df.groupby("item")["sales"].mean()
    preds = test_df["item"].map(item_mean).fillna(train_df["sales"].mean())
    rmse = mean_squared_error(test_df["sales"], preds)**0.5
    return rmse


# ------------ LightGBM Training ------------
def train_lgbm(X_train, y_train):
    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="regression",
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


# ------------ Random Search Tuning ------------
def random_search_lgbm(X_train, y_train, X_val, y_val, n_iter=10):
    best_params, best_rmse = None, float("inf")

    for i in range(n_iter):
        params = {
            "n_estimators": random.choice([300, 500, 800]),
            "learning_rate": random.choice([0.01, 0.03, 0.05, 0.1]),
            "max_depth": random.choice([-1, 8, 10, 12]),
            "num_leaves": random.choice([31, 63, 127]),
            "subsample": random.choice([0.7, 0.8, 1.0]),
            "colsample_bytree": random.choice([0.7, 0.8, 1.0]),
            "min_child_samples": random.choice([20, 50, 100]),
            "objective": "regression",
            "random_state": 42
        }

        model = LGBMRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        rmse = mean_squared_error(y_val, preds)**0.5
        print(f"Iter {i+1}/{n_iter} - RMSE {rmse:.4f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_params = params

    return best_params, best_rmse


# ------------ Feature Importance Plot ------------
def plot_feature_importance(model, feature_cols, out_path="plots/feature_importance.png"):
    importances = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    plt.figure(figsize=(8, 6))
    sns.barplot(data=importances.head(15), x="importance", y="feature")
    plt.title("Top 15 Feature Importances (LightGBM)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
