# feature_engineering.py
# ----------------------------------------
# Functions for creating time-series features:
# - Date/time features
# - Lag features
# - Rolling window statistics
# ----------------------------------------

import pandas as pd

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar/time-based features."""
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dayofweek"] = df["date"].dt.dayofweek
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    return df


def add_lag_features(df: pd.DataFrame,
                     group_cols=("store", "item"),
                     target="sales") -> pd.DataFrame:
    """Add lag features and rolling window statistics."""

    # Basic lags
    for lag in [1, 2, 3, 7, 14, 30]:
        df[f"lag_{lag}"] = (
            df.groupby(list(group_cols))[target]
              .shift(lag)
        )

    # Rolling means
    df["roll_mean_7"] = (
        df.groupby(list(group_cols))[target]
          .shift(1).rolling(window=7).mean()
    )
    df["roll_mean_30"] = (
        df.groupby(list(group_cols))[target]
          .shift(1).rolling(window=30).mean()
    )

    # Rolling std
    df["roll_std_7"] = (
        df.groupby(list(group_cols))[target]
          .shift(1).rolling(window=7).std()
    )

    return df
