"""
feature_engineering.py
----------------------
Constructs historical lag features, moving averages, percentage changes, and the
next-session target variable (`Target_Next_Close`) with strict prevention of data leakage.

Data Leakage Prevention Rules Enforced:
1. Every input feature at observation index `t` is computed strictly from prices at or before `t` (<= t).
2. Rolling windows (`MA_7`, `MA_14`, `Volatility_7`) are right-aligned at `t` (never centered).
3. The target variable (`Target_Next_Close`) is the next available trading session's closing price (`t + 1`).
4. The final row of the dataset (`t = latest date`) has all input features populated so it can
   predict the upcoming trading session (`t + 1`) without needing future target values.
"""

from typing import Tuple, List, Dict, Any
import pandas as pd
import numpy as np


FEATURE_COLUMNS: List[str] = [
    "Current_Close",   # Latest available trading session price (P_t)
    "Lag_1",           # Previous 1 trading session price (P_{t-1})
    "Lag_2",           # Previous 2 trading sessions price (P_{t-2})
    "Lag_3",           # Previous 3 trading sessions price (P_{t-3})
    "Lag_7",           # Previous 7 trading sessions price (P_{t-7})
    "MA_7",            # 7-session simple moving average up to session t
    "MA_14",           # 14-session simple moving average up to session t
    "Pct_Change",      # 1-session historical percentage change (%) at session t
    "Pct_Change_7",    # 7-session historical percentage change (%) at session t
    "Volatility_7",    # 7-session rolling standard deviation of daily % returns
]

TARGET_COLUMN = "Target_Next_Close"

FEATURE_DESCRIPTIONS: Dict[str, str] = {
    "Current_Close": "Latest available closing price at session t (1 session before target t+1)",
    "Lag_1": "Gold closing price 1 trading session prior (t - 1)",
    "Lag_2": "Gold closing price 2 trading sessions prior (t - 2)",
    "Lag_3": "Gold closing price 3 trading sessions prior (t - 3)",
    "Lag_7": "Gold closing price 7 trading sessions prior (t - 7)",
    "MA_7": "7-trading-session Rolling Moving Average (t - 6 through t)",
    "MA_14": "14-trading-session Rolling Moving Average (t - 13 through t)",
    "Pct_Change": "1-session historical price percentage change (%)",
    "Pct_Change_7": "7-session historical price percentage change (%)",
    "Volatility_7": "7-session rolling standard deviation of daily percentage changes",
}


def create_features(clean_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Create historical lag features, moving averages, return features, and the
    supervised target variable (`Target_Next_Close`).

    Returns:
    --------
    supervised_df : pd.DataFrame
        Rows where both historical features and `Target_Next_Close` are valid and non-null.
        Used for chronological train-test splitting and model evaluation.
    full_feature_df : pd.DataFrame
        All rows where historical input features are valid (including the very last trading day,
        whose `Target_Next_Close` is NaN because it lies in the future).
    latest_feature_row : pd.Series
        The single latest observation row (`t = max(Date)`) used to predict the next trading session.
    """
    if clean_df is None or clean_df.empty:
        raise ValueError("Cannot engineer features from an empty DataFrame.")

    df = clean_df.sort_values("Date", ascending=True).reset_index(drop=True).copy()

    # Latest known price at prediction time t
    df["Current_Close"] = df["Close"]

    # Historical Lag Features (strictly looking backward in time)
    df["Lag_1"] = df["Close"].shift(1)
    df["Lag_2"] = df["Close"].shift(2)
    df["Lag_3"] = df["Close"].shift(3)
    df["Lag_7"] = df["Close"].shift(7)

    # Causal Rolling Moving Averages (right-aligned window ending at t; center=False)
    df["MA_7"] = df["Close"].rolling(window=7, min_periods=7, center=False).mean()
    df["MA_14"] = df["Close"].rolling(window=14, min_periods=14, center=False).mean()

    # Historical Percentage Changes & Short-Term Volatility
    df["Pct_Change"] = df["Close"].pct_change(periods=1) * 100.0
    df["Pct_Change_7"] = df["Close"].pct_change(periods=7) * 100.0
    df["Volatility_7"] = df["Pct_Change"].rolling(window=7, min_periods=7, center=False).std()

    # Target Variable: Next available trading session's closing price (P_{t+1})
    df[TARGET_COLUMN] = df["Close"].shift(-1)
    df["Target_Date"] = df["Date"].shift(-1)

    # Filter out initial warmup rows (first 13 rows where MA_14 / Lag_7 are NaN)
    full_feature_df = df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    if full_feature_df.empty:
        raise ValueError("Not enough rows remain after computing 14-session rolling features.")

    # Latest observation row (has all features, Target_Next_Close is NaN)
    latest_feature_row = full_feature_df.iloc[-1].copy()

    # Supervised dataset (all rows that also have a known ground-truth Target_Next_Close)
    supervised_df = full_feature_df.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)

    return supervised_df, full_feature_df, latest_feature_row

