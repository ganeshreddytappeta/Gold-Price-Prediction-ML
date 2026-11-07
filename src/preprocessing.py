"""
preprocessing.py
----------------
Handles chronological sorting, timestamp validation, duplicate date removal,
and invalid price filtering without fabricating or interpolating missing market prices.
Also provides Weekly and Monthly resampled views for historical trend analysis.
"""

from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np


class PreprocessingError(Exception):
    """Custom exception raised when data preprocessing fails."""
    pass


MIN_REQUIRED_ROWS = 50


def preprocess_gold_data(
    raw_df: pd.DataFrame,
    date_col: str,
    price_col: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean and standardize the raw gold dataset while preventing data fabrication.

    Steps:
    1. Standardize date column to datetime and remove rows with invalid dates.
    2. Coerce price column to numeric and drop missing or non-positive (<= 0) prices.
       (Never invent or interpolate fake market prices.)
    3. Sort strictly in ascending chronological order.
    4. Remove duplicate dates (keeping the latest record if duplicates exist).
    5. Standardize core column aliases (`Date` and `Close`) while preserving existing columns.
    """
    if raw_df is None or raw_df.empty:
        raise PreprocessingError("Cannot preprocess an empty DataFrame.")

    if date_col not in raw_df.columns:
        raise PreprocessingError(f"Date column '{date_col}' not found in dataset.")
    if price_col not in raw_df.columns:
        raise PreprocessingError(f"Price column '{price_col}' not found in dataset.")

    df = raw_df.copy()
    initial_rows = len(df)

    # 1. Parse dates
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    invalid_date_count = int(df[date_col].isna().sum())
    if invalid_date_count > 0:
        df = df.dropna(subset=[date_col]).copy()

    # 2. Validate numeric positive prices (never fabricate missing prices)
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    invalid_price_mask = df[price_col].isna() | (df[price_col] <= 0) | np.isinf(df[price_col])
    invalid_price_count = int(invalid_price_mask.sum())
    if invalid_price_count > 0:
        df = df.loc[~invalid_price_mask].copy()

    # 3. Sort chronologically (oldest to newest)
    df = df.sort_values(by=date_col, ascending=True).reset_index(drop=True)

    # 4. Remove duplicate dates
    duplicate_date_count = int(df.duplicated(subset=[date_col], keep="last").sum())
    if duplicate_date_count > 0:
        df = df.drop_duplicates(subset=[date_col], keep="last").reset_index(drop=True)

    # 5. Create standardized column references 'Date' and 'Close' if named differently
    if date_col != "Date":
        df["Date"] = df[date_col]
    if price_col != "Close":
        df["Close"] = df[price_col]

    # Ensure optional OHLCV columns are numeric if present
    for optional_col in ["Open", "High", "Low", "Volume", "Adj Close"]:
        if optional_col in df.columns:
            df[optional_col] = pd.to_numeric(df[optional_col], errors="coerce")

    final_rows = len(df)
    if final_rows < MIN_REQUIRED_ROWS:
        raise PreprocessingError(
            f"Insufficient valid historical observations ({final_rows} rows). "
            f"At least {MIN_REQUIRED_ROWS} trading sessions are required for reliable feature engineering and training."
        )

    preprocessing_summary: Dict[str, Any] = {
        "initial_rows": initial_rows,
        "final_rows": final_rows,
        "dropped_invalid_dates": invalid_date_count,
        "dropped_invalid_prices": invalid_price_count,
        "dropped_duplicate_dates": duplicate_date_count,
        "clean_start_date": df["Date"].min().strftime("%Y-%m-%d"),
        "clean_end_date": df["Date"].max().strftime("%Y-%m-%d"),
    }

    return df, preprocessing_summary


def get_resampled_trends(clean_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Generate Daily (trading session), Weekly, and Monthly aggregated views
    from the cleaned historical DataFrame for Section B (Historical Analysis).
    """
    if clean_df is None or clean_df.empty:
        raise PreprocessingError("Clean DataFrame is empty; cannot compute trend views.")

    base = clean_df[["Date", "Close"]].copy()
    for col in ["Open", "High", "Low", "Volume"]:
        if col in clean_df.columns:
            base[col] = clean_df[col]

    base = base.sort_values("Date").set_index("Date")

    # Daily view (actual trading sessions without fabricating weekend rows)
    daily_df = base.reset_index()

    # Weekly view (week ending Friday)
    agg_rules: Dict[str, Any] = {"Close": "last"}
    if "Open" in base.columns:
        agg_rules["Open"] = "first"
    if "High" in base.columns:
        agg_rules["High"] = "max"
    if "Low" in base.columns:
        agg_rules["Low"] = "min"
    if "Volume" in base.columns:
        agg_rules["Volume"] = "sum"

    weekly_df = base.resample("W-FRI").agg(agg_rules).dropna(subset=["Close"]).reset_index()
    weekly_df["Weekly_Return_Pct"] = weekly_df["Close"].pct_change() * 100.0

    # Monthly view (month end)
    try:
        monthly_df = base.resample("ME").agg(agg_rules).dropna(subset=["Close"]).reset_index()
    except ValueError:
        monthly_df = base.resample("M").agg(agg_rules).dropna(subset=["Close"]).reset_index()
    monthly_df["Monthly_Return_Pct"] = monthly_df["Close"].pct_change() * 100.0

    return {
        "Daily": daily_df,
        "Weekly": weekly_df,
        "Monthly": monthly_df,
    }

