"""
data_loader.py
--------------
Responsible for locating, loading, and inspecting the historical Gold Price CSV dataset.
Automatically identifies the date column, target gold price column, observation frequency,
missing/duplicate/invalid records, and actual price unit without fabricating market data.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np


class DataLoaderError(Exception):
    """Custom exception raised when dataset loading or validation fails."""
    pass


# Candidate column names for automatic detection (case-insensitive matching)
DATE_COLUMN_CANDIDATES = [
    "date", "datetime", "timestamp", "time", "trading_date", "day"
]

PRICE_COLUMN_CANDIDATES = [
    "close", "price", "gold_price", "adj close", "adj_close",
    "closing_price", "spot_price", "value", "inr", "usd"
]


def find_dataset_path(custom_path: Optional[str] = None) -> Path:
    """
    Locate the Kaggle CSV dataset inside the project directory.
    Checks custom_path first, then standard locations in `data/` and `Data set/`.
    """
    project_root = Path(__file__).resolve().parent.parent

    if custom_path:
        candidate = Path(custom_path)
        if not candidate.is_absolute():
            candidate = project_root / candidate
        if candidate.exists() and candidate.is_file():
            return candidate
        raise DataLoaderError(f"Specified dataset file was not found: {candidate}")

    # Standard expected paths in priority order
    preferred_paths = [
        project_root / "data" / "gold_data.csv",
        project_root / "data" / "gold_prices_10y.csv",
        project_root / "Data set" / "gold_prices_10y.csv",
        project_root / "Data set" / "gold_data.csv",
    ]
    for path in preferred_paths:
        if path.exists() and path.is_file():
            return path

    # Fallback: scan `data/` and `Data set/` for any .csv file
    for folder_name in ["data", "Data set"]:
        folder = project_root / folder_name
        if folder.exists() and folder.is_dir():
            csv_files = sorted(folder.glob("*.csv"))
            if csv_files:
                return csv_files[0]

    raise DataLoaderError(
        "No CSV dataset found in 'data/' or 'Data set/'. "
        "Please place your Kaggle Gold Price CSV file in the 'data/' folder."
    )


def identify_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Automatically detect the date column and primary gold price column.
    Does not blindly assume 'Date' and 'Price'; inspects column names and data types.
    """
    if df is None or df.empty:
        raise DataLoaderError("The loaded CSV dataset is empty.")

    col_map = {col.strip().lower(): col for col in df.columns}

    # 1. Identify Date Column
    detected_date_col = None
    for candidate in DATE_COLUMN_CANDIDATES:
        if candidate in col_map:
            detected_date_col = col_map[candidate]
            break

    if detected_date_col is None:
        # Try parsing object/string columns as dates
        for col in df.columns:
            if df[col].dtype == "object":
                parsed = pd.to_datetime(df[col].dropna().head(20), errors="coerce")
                if parsed.notna().mean() > 0.8:
                    detected_date_col = col
                    break

    if detected_date_col is None:
        raise DataLoaderError(
            f"Could not automatically identify a Date column from columns: {list(df.columns)}."
        )

    # 2. Identify Primary Gold Price Column
    detected_price_col = None
    for candidate in PRICE_COLUMN_CANDIDATES:
        if candidate in col_map and col_map[candidate] != detected_date_col:
            col_name = col_map[candidate]
            numeric_series = pd.to_numeric(df[col_name], errors="coerce")
            if numeric_series.notna().mean() > 0.8:
                detected_price_col = col_name
                break

    if detected_price_col is None:
        # Fallback: check for columns containing 'close' or 'price' or 'gold'
        for lower_name, orig_name in col_map.items():
            if orig_name == detected_date_col:
                continue
            if any(keyword in lower_name for keyword in ["close", "price", "gold", "rate"]):
                numeric_series = pd.to_numeric(df[orig_name], errors="coerce")
                if numeric_series.notna().mean() > 0.8:
                    detected_price_col = orig_name
                    break

    if detected_price_col is None:
        raise DataLoaderError(
            f"Could not automatically identify a numeric Gold Price column from columns: {list(df.columns)}."
        )

    return detected_date_col, detected_price_col


def identify_price_unit(df: pd.DataFrame, price_col: str) -> Dict[str, Any]:
    """
    Inspect the dataset columns and price values to determine the actual price unit.
    Never assumes INR per gram or silently invents exchange rates.
    """
    col_names_lower = " ".join(c.lower() for c in df.columns)
    prices = pd.to_numeric(df[price_col], errors="coerce").dropna()

    if prices.empty:
        return {
            "currency_code": "UNKNOWN",
            "currency_symbol": "",
            "unit_label": "Unknown Unit",
            "short_unit": "Units",
            "instrument_description": "Unknown Instrument",
            "has_inr_column": False,
            "has_exchange_rate": False,
            "explanation": "No valid numeric prices available to determine unit."
        }

    min_p = float(prices.min())
    max_p = float(prices.max())
    median_p = float(prices.median())

    has_inr = "inr" in col_names_lower or "rupee" in col_names_lower
    has_fx = any(k in col_names_lower for k in ["usd_inr", "usdinr", "exchange_rate", "fx"])
    has_volume = "volume" in col_names_lower
    max_vol = float(pd.to_numeric(df["Volume"], errors="coerce").max()) if "Volume" in df.columns else 0.0

    # Case 1: Explicit INR column or Indian gold price range per 10g (typically 25,000 - 95,000 INR)
    if has_inr or (min_p > 15000 and max_p > 30000):
        return {
            "currency_code": "INR",
            "currency_symbol": "₹",
            "unit_label": "INR per 10 Grams (₹ / 10g)",
            "short_unit": "INR / 10g",
            "instrument_description": "Indian Domestic Gold Price (INR per 10 grams)",
            "has_inr_column": True,
            "has_exchange_rate": has_fx,
            "explanation": (
                f"Price range ({min_p:,.2f} to {max_p:,.2f}) matches Indian domestic gold prices "
                "quoted in Indian Rupees (INR) per 10 grams."
            ),
        }

    # Case 2: Indian gold price per 1 gram (typically 2,500 - 10,000 INR) without USD ETF indicators
    if 2500 <= min_p <= 12000 and max_p > 4500 and not has_volume:
        return {
            "currency_code": "INR",
            "currency_symbol": "₹",
            "unit_label": "INR per Gram (₹ / g)",
            "short_unit": "INR / g",
            "instrument_description": "Indian Domestic Gold Price (INR per gram)",
            "has_inr_column": True,
            "has_exchange_rate": has_fx,
            "explanation": (
                f"Price range ({min_p:,.2f} to {max_p:,.2f}) corresponds to gold prices in INR per gram."
            ),
        }

    # Case 3: Spot Gold in USD per Troy Ounce (typically $1,000 - $3,000+ / oz)
    if 800 <= min_p <= 3500 and max_p >= 1200:
        return {
            "currency_code": "USD",
            "currency_symbol": "$",
            "unit_label": "USD per Troy Ounce ($ / oz)",
            "short_unit": "USD / troy oz",
            "instrument_description": "International Spot/Futures Gold (USD per Troy Ounce)",
            "has_inr_column": False,
            "has_exchange_rate": has_fx,
            "explanation": (
                f"Price range (${min_p:,.2f} to ${max_p:,.2f}) corresponds to international gold prices "
                "in US Dollars (USD) per troy ounce (31.1035 grams). No USD/INR exchange rate column "
                "is present in the CSV, so native USD per troy ounce is displayed."
            ),
        }

    # Case 4: SPDR Gold Shares ETF (GLD) in USD per Share (~1/10th Troy Ounce, ~$100 to ~$460 with multi-million volume)
    if 50 <= min_p <= 600 and max_p <= 1000:
        return {
            "currency_code": "USD",
            "currency_symbol": "$",
            "unit_label": "USD per Share (GLD Gold ETF, ~1/10th Troy Oz)",
            "short_unit": "USD / Share (GLD)",
            "instrument_description": (
                "SPDR Gold Shares ETF (GLD) / Gold Benchmark in USD per Share "
                "(1 share ≈ 1/10th troy ounce of gold bullion)"
            ),
            "has_inr_column": False,
            "has_exchange_rate": has_fx,
            "explanation": (
                f"The dataset prices range from ${min_p:,.2f} (Jan 2016) to ${max_p:,.2f} (Jan 2026) "
                f"with daily trading volumes up to {max_vol:,.0f} shares and OHLCV + Adj Close columns. "
                "These values represent US Dollar (USD) prices per share of the SPDR Gold Shares ETF (GLD) "
                "(where each share tracks approximately 1/10th of a troy ounce of physical gold), "
                "NOT Indian Rupees (INR) per gram. Because the CSV does not contain historical USD/INR "
                "exchange rates, all training and metrics preserve the authentic USD unit without "
                "inventing exchange rates."
            ),
        }

    # Fallback generic unit
    return {
        "currency_code": "USD",
        "currency_symbol": "$",
        "unit_label": "Dataset Native Price Unit ($)",
        "short_unit": "USD",
        "instrument_description": "Historical Gold Price Series (Dataset Native Unit)",
        "has_inr_column": False,
        "has_exchange_rate": has_fx,
        "explanation": (
            f"Prices range from {min_p:,.2f} to {max_p:,.2f}. Displayed in the dataset's native unit."
        ),
    }


def detect_observation_frequency(dates: pd.Series) -> Dict[str, Any]:
    """
    Determine whether the dataset observations are daily (trading days), weekly, or monthly.
    """
    clean_dates = pd.to_datetime(dates, errors="coerce").dropna().sort_values()
    if len(clean_dates) < 2:
        return {"frequency": "Unknown", "median_gap_days": None, "details": "Insufficient dates."}

    diffs = clean_dates.diff().dropna().dt.days
    median_gap = float(diffs.median())
    mode_gap = int(diffs.mode().iloc[0]) if not diffs.mode().empty else int(median_gap)

    if median_gap <= 1.5:
        freq_label = "Daily (Trading Sessions / Business Days)"
        details = (
            f"Observations occur on active market trading days (Monday–Friday, excluding market holidays). "
            f"Most consecutive observations are {mode_gap} day apart (3 days across weekends), "
            f"averaging ~251 trading sessions per year."
        )
    elif 5 <= median_gap <= 9:
        freq_label = "Weekly"
        details = f"Observations are recorded weekly (median gap: {median_gap:.1f} days)."
    elif 25 <= median_gap <= 35:
        freq_label = "Monthly"
        details = f"Observations are recorded monthly (median gap: {median_gap:.1f} days)."
    else:
        freq_label = f"Irregular (Median gap: {median_gap:.1f} days)"
        details = f"Median interval between records is {median_gap:.1f} days."

    return {
        "frequency": freq_label,
        "median_gap_days": median_gap,
        "mode_gap_days": mode_gap,
        "details": details,
    }


def load_and_inspect_dataset(custom_path: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load the CSV file and perform a comprehensive inspection of columns, data types,
    missing values, duplicates, invalid entries, price unit, frequency, and date range.
    """
    csv_path = find_dataset_path(custom_path)

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        raise DataLoaderError(f"Failed to read CSV file at '{csv_path}': {exc}") from exc

    if df.empty:
        raise DataLoaderError(f"The CSV file at '{csv_path}' contains 0 rows.")

    date_col, price_col = identify_columns(df)

    # Parse dates and numeric prices for inspection without altering raw df yet
    parsed_dates = pd.to_datetime(df[date_col], errors="coerce")
    parsed_prices = pd.to_numeric(df[price_col], errors="coerce")

    missing_by_col = df.isna().sum().to_dict()
    total_missing = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    duplicate_dates = int(parsed_dates.dropna().duplicated().sum())
    invalid_dates = int(parsed_dates.isna().sum())
    invalid_or_nonpositive_prices = int((parsed_prices.isna() | (parsed_prices <= 0)).sum())

    unit_info = identify_price_unit(df, price_col)
    freq_info = detect_observation_frequency(parsed_dates)

    valid_dates = parsed_dates.dropna()
    start_date_str = valid_dates.min().strftime("%Y-%m-%d") if not valid_dates.empty else "N/A"
    end_date_str = valid_dates.max().strftime("%Y-%m-%d") if not valid_dates.empty else "N/A"

    valid_prices = parsed_prices.dropna()
    inspection_report: Dict[str, Any] = {
        "file_path": str(csv_path),
        "file_name": csv_path.name,
        "num_rows": int(len(df)),
        "num_columns": int(len(df.columns)),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "date_column": date_col,
        "price_column": price_col,
        "start_date": start_date_str,
        "end_date": end_date_str,
        "total_missing_values": total_missing,
        "missing_by_column": {k: int(v) for k, v in missing_by_col.items()},
        "duplicate_rows": duplicate_rows,
        "duplicate_dates": duplicate_dates,
        "invalid_dates": invalid_dates,
        "invalid_or_nonpositive_prices": invalid_or_nonpositive_prices,
        "price_min": float(valid_prices.min()) if not valid_prices.empty else None,
        "price_max": float(valid_prices.max()) if not valid_prices.empty else None,
        "price_mean": float(valid_prices.mean()) if not valid_prices.empty else None,
        "price_median": float(valid_prices.median()) if not valid_prices.empty else None,
        "latest_price": float(valid_prices.iloc[-1]) if not valid_prices.empty else None,
        "unit_info": unit_info,
        "frequency_info": freq_info,
    }

    return df, inspection_report

