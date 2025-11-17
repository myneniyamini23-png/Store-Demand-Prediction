# walk_forward_validation.py
# --------------------------------------------------------
# Function for walk-forward (rolling) validation
# --------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor


def walk_forward_eval(df, features, params, target="sales",
                      n_splits=3, min_train_days=365*2, val_days=90):

    dates = df["date"].sort_values().unique()
    results = []
    start_idx = min_train_days

    for i in range(n_splits):
        train_end_date = dates[start_idx]
        val_end_idx = min(start_idx + val_days, len(dates) - 1)
        val_end_date = dates[val_end_idx]

        train_mask = df["date"] <= train_end_date
        val_mask = (df["date"] > train_end_date) & (df["date"] <= val_end_date)

        train_fold = df[train_mask]
        val_fold = df[val_mask]

        if len(val_fold) == 0:
            break

        X_train = train_fold[features]
        y_train = train_fold[target]
        X_val = val_fold[features]
        y_val = val_fold[target]

        model = LGBMRegressor(**params)
        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        rmse = mean_squared_error(y_val, preds)**0.5

        results.append({
            "fold": i+1,
            "train_end": str(train_end_date.date()),
            "val_end": str(val_end_date.date()),
            "rmse": rmse
        })

        print(f"[Fold {i+1}] RMSE={rmse:.4f}")

        start_idx = val_end_idx
        if val_end_idx == len(dates) - 1:
            break

    return pd.DataFrame(results)
