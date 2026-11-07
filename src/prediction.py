"""
prediction.py
-------------
Loads the saved Joblib model artifacts and generates:
1. Next trading session's gold price prediction using the latest historical observation.
2. Multi-step recursive forecasts for any custom future date up to 365 calendar days ahead.
Avoids retraining on every page refresh while detecting dataset changes.
"""

from datetime import date, datetime
from typing import Dict, Any, Optional, Union, List
import joblib
import numpy as np
import pandas as pd

from src.data_loader import load_and_inspect_dataset, find_dataset_path
from src.preprocessing import preprocess_gold_data
from src.feature_engineering import create_features
from src.train_model import (
    BEST_MODEL_PATH,
    METADATA_PATH,
    compute_file_hash,
    train_and_evaluate_models,
)


class PredictionError(Exception):
    """Custom exception raised when loading models or generating predictions fails."""
    pass


MAX_FORECAST_CALENDAR_DAYS = 365


def load_or_train_artifacts(
    custom_data_path: Optional[str] = None,
    force_retrain: bool = False,
) -> Dict[str, Any]:
    """
    Load saved model metadata from disk if available and up-to-date with the CSV dataset.
    Retrains automatically only if saved models are missing, corrupted, dataset changed,
    or `force_retrain=True` is explicitly requested.
    """
    csv_path = find_dataset_path(custom_data_path)
    current_hash = compute_file_hash(str(csv_path))

    if not force_retrain and BEST_MODEL_PATH.exists() and METADATA_PATH.exists():
        try:
            metadata = joblib.load(METADATA_PATH)
            saved_hash = metadata.get("dataset_hash")
            if saved_hash == current_hash:
                return metadata
        except Exception:
            # If saved model file is corrupted or incompatible, fall through to retrain
            pass

    try:
        metadata = train_and_evaluate_models(custom_data_path=str(csv_path))
        return metadata
    except Exception as exc:
        raise PredictionError(f"Failed to train or load prediction models: {exc}") from exc


def predict_next_trading_session(
    custom_data_path: Optional[str] = None,
    selected_model_name: Optional[str] = None,
    force_retrain: bool = False,
) -> Dict[str, Any]:
    """
    Generate the prediction for the next available trading session following the
    latest date in the dataset.
    """
    metadata = load_or_train_artifacts(
        custom_data_path=custom_data_path,
        force_retrain=force_retrain,
    )

    raw_df, inspection_report = load_and_inspect_dataset(custom_data_path)
    clean_df, _ = preprocess_gold_data(
        raw_df,
        inspection_report["date_column"],
        inspection_report["price_column"],
    )
    _, full_feature_df, latest_feature_row = create_features(clean_df)

    feature_cols = metadata["feature_columns"]
    scaler = metadata["scaler"]
    trained_models = metadata["trained_models"]
    best_model_name = metadata["best_model_name"]

    active_model_name = selected_model_name if (
        selected_model_name and selected_model_name in trained_models
    ) else best_model_name

    # Prepare single-row input vector for latest observation t
    X_latest_df = pd.DataFrame([latest_feature_row[feature_cols].values], columns=feature_cols)
    X_latest_scaled = scaler.transform(X_latest_df)
    X_latest_raw = X_latest_df.values

    latest_date_val = pd.to_datetime(latest_feature_row["Date"])
    latest_date_str = latest_date_val.strftime("%Y-%m-%d")
    latest_day_name = latest_date_val.strftime("%A")
    latest_price = float(latest_feature_row["Current_Close"])

    # Compute predictions from all 4 trained models for comparison
    all_model_predictions: Dict[str, Dict[str, float]] = {}
    for m_name, m_bundle in trained_models.items():
        estimator = m_bundle["model"]
        use_scaled = m_bundle["use_scaled"]
        X_in = X_latest_scaled if use_scaled else X_latest_raw
        pred_val = float(estimator.predict(X_in)[0])
        abs_diff = pred_val - latest_price
        pct_diff = (abs_diff / latest_price) * 100.0 if latest_price != 0 else 0.0
        all_model_predictions[m_name] = {
            "predicted_price": pred_val,
            "absolute_change": abs_diff,
            "percentage_change": pct_diff,
        }

    # Also include Persistence Baseline
    all_model_predictions["Persistence Baseline (Naive)"] = {
        "predicted_price": latest_price,
        "absolute_change": 0.0,
        "percentage_change": 0.0,
    }

    primary_pred_info = all_model_predictions[active_model_name]

    return {
        "latest_date": latest_date_str,
        "latest_day_name": latest_day_name,
        "latest_price": latest_price,
        "predicted_price": primary_pred_info["predicted_price"],
        "absolute_change": primary_pred_info["absolute_change"],
        "percentage_change": primary_pred_info["percentage_change"],
        "best_model_name": best_model_name,
        "active_model_name": active_model_name,
        "all_model_predictions": all_model_predictions,
        "latest_features": {col: float(latest_feature_row[col]) for col in feature_cols},
        "unit_info": inspection_report["unit_info"],
        "metadata": metadata,
        "clean_df": clean_df,
        "full_feature_df": full_feature_df,
    }


def _build_step_feature_vector(price_history: List[float], feature_cols: List[str]) -> pd.DataFrame:
    """
    Construct the 10 causal historical features from the rolling `price_history` list
    (which contains at least 15 prices: actual historical prices followed by prior recursive predictions).
    Never uses future actual prices.
    """
    if len(price_history) < 15:
        raise PredictionError("At least 15 historical prices are required to compute 14-session rolling features.")

    current_close = float(price_history[-1])
    lag_1 = float(price_history[-2])
    lag_2 = float(price_history[-3])
    lag_3 = float(price_history[-4])
    lag_7 = float(price_history[-8])

    ma_7 = float(np.mean(price_history[-7:]))
    ma_14 = float(np.mean(price_history[-14:]))

    pct_change = ((current_close - lag_1) / lag_1) * 100.0 if lag_1 != 0 else 0.0
    pct_change_7 = ((current_close - lag_7) / lag_7) * 100.0 if lag_7 != 0 else 0.0

    # Compute the last 7 daily percentage changes for Volatility_7 (matching pandas .rolling(7).std(ddof=1))
    recent_8 = np.array(price_history[-8:], dtype=float)
    recent_7_returns = ((recent_8[1:] - recent_8[:-1]) / recent_8[:-1]) * 100.0
    volatility_7 = float(np.std(recent_7_returns, ddof=1))

    feature_map = {
        "Current_Close": current_close,
        "Lag_1": lag_1,
        "Lag_2": lag_2,
        "Lag_3": lag_3,
        "Lag_7": lag_7,
        "MA_7": ma_7,
        "MA_14": ma_14,
        "Pct_Change": pct_change,
        "Pct_Change_7": pct_change_7,
        "Volatility_7": volatility_7,
    }

    row_values = [feature_map[col] for col in feature_cols]
    return pd.DataFrame([row_values], columns=feature_cols)


def _get_horizon_uncertainty_assessment(
    calendar_days: int,
    trading_sessions: int,
    is_weekend_target: bool,
    selected_date_str: str,
    effective_trading_date_str: str,
) -> Dict[str, str]:
    """
    Generate a transparent, beginner-friendly uncertainty assessment and warning
    tailored to the user's selected forecast horizon.
    """
    weekend_note = ""
    if is_weekend_target:
        weekend_note = (
            f" Note: Your selected date ({selected_date_str}) falls on a weekend when gold markets are closed; "
            f"the forecast corresponds to the active trading session on {effective_trading_date_str}."
        )

    if trading_sessions == 1:
        level = "Low–Moderate (1-Step Direct Forecast)"
        badge_color = "info"
        warning_text = (
            f"This forecast is for the **next available trading session** ({trading_sessions} trading session / "
            f"{calendar_days} calendar day{'s' if calendar_days != 1 else ''} ahead). All input features come "
            f"100% from observed historical market data with zero recursive feedback.{weekend_note}"
        )
    elif calendar_days <= 14:
        level = "Moderate (Short-Term Recursive Forecast)"
        badge_color = "warning"
        warning_text = (
            f"**Short-Term Multi-Step Forecast ({calendar_days} calendar days / ~{trading_sessions} trading sessions ahead):** "
            f"Predictions beyond the first session are generated using **recursive forecasting**, where each day's "
            f"predicted price is fed back as `Current_Close` and `Lag` features for subsequent days. Small prediction "
            f"errors compound across steps.{weekend_note}"
        )
    elif calendar_days <= 60:
        level = "Elevated (Medium-Term Recursive Forecast)"
        badge_color = "warning"
        warning_text = (
            f"**Medium-Term Forecast Uncertainty Warning ({calendar_days} calendar days / ~{trading_sessions} trading sessions ahead):** "
            f"After 14 trading sessions, every lag (`Lag_1`–`Lag_7`) and moving average (`MA_7`, `MA_14`) input inside "
            f"the model consists entirely of previous model predictions rather than real market observations. "
            f"Forecast uncertainty widens significantly over multi-week horizons.{weekend_note}"
        )
    else:
        level = "High Uncertainty (Long-Term Extrapolated Forecast)"
        badge_color = "error"
        warning_text = (
            f"**⚠️ High Long-Term Uncertainty Warning ({calendar_days} calendar days / ~{trading_sessions} trading sessions ahead):** "
            f"Forecasting several months or up to a year ahead relies on {trading_sessions} sequential recursive steps. "
            f"Long-term gold prices are driven by macroeconomic shifts (central bank interest rates, inflation, geopolitical "
            f"events, and currency movements) that cannot be predicted from historical price lags alone. Treat this figure "
            f"strictly as a statistical trend projection, **not** a high-accuracy price guarantee.{weekend_note}"
        )

    return {
        "uncertainty_level": level,
        "badge_color": badge_color,
        "warning_message": warning_text,
    }


def predict_custom_future_date(
    target_date: Union[str, date, datetime, pd.Timestamp],
    custom_data_path: Optional[str] = None,
    selected_model_name: Optional[str] = None,
    force_retrain: bool = False,
) -> Dict[str, Any]:
    """
    Predict the estimated gold price for any user-selected future date up to 365 calendar days
    after the latest available observation in the dataset.

    Forecasting Strategy:
    - For 1 trading session ahead: uses the exact 1-step next-day prediction model on actual historical features.
    - For multiple trading sessions ahead (h = 2 ... N): applies sequential **Recursive Multi-Step Forecasting**:
      At each estimated future trading day (Monday–Friday business calendar), the model predicts the closing
      price, appends it to the chronological price series, and recomputes all lag (`Lag_1`, `Lag_2`, `Lag_3`, `Lag_7`),
      moving-average (`MA_7`, `MA_14`), and return (`Pct_Change`, `Pct_Change_7`, `Volatility_7`) features
      strictly from the updated series.
    """
    base_pipeline = predict_next_trading_session(
        custom_data_path=custom_data_path,
        selected_model_name=selected_model_name,
        force_retrain=force_retrain,
    )

    metadata = base_pipeline["metadata"]
    clean_df = base_pipeline["clean_df"]
    latest_date_ts = pd.to_datetime(base_pipeline["latest_date"]).normalize()
    latest_price = float(base_pipeline["latest_price"])
    best_model_name = base_pipeline["best_model_name"]
    active_model_name = base_pipeline["active_model_name"]

    # Validate and normalize target_date
    try:
        target_ts = pd.to_datetime(target_date).normalize()
    except Exception as exc:
        raise PredictionError(f"Invalid target date '{target_date}': {exc}") from exc

    if pd.isna(target_ts):
        raise PredictionError("Selected target date cannot be empty or NaT.")

    calendar_days = int((target_ts - latest_date_ts).days)
    if calendar_days <= 0:
        raise PredictionError(
            f"Past or current historical dates are not allowed. Please select a future date strictly after "
            f"the latest available dataset observation ({latest_date_ts.strftime('%Y-%m-%d')})."
        )

    if calendar_days > MAX_FORECAST_CALENDAR_DAYS:
        raise PredictionError(
            f"Selected date ({target_ts.strftime('%Y-%m-%d')}) is {calendar_days} days ahead, which exceeds "
            f"the maximum supported horizon of {MAX_FORECAST_CALENDAR_DAYS} calendar days."
        )

    # Determine estimated trading sessions (Monday-Friday business calendar)
    start_bdate = latest_date_ts + pd.Timedelta(days=1)
    bdates = pd.bdate_range(start=start_bdate, end=target_ts)

    is_weekend_target = target_ts.weekday() >= 5
    if len(bdates) == 0:
        # Target date is the immediate Saturday/Sunday after a Friday observation;
        # forecast the next available trading session (Monday)
        next_bday = latest_date_ts + pd.tseries.offsets.BDay(1)
        bdates = pd.DatetimeIndex([next_bday])

    effective_trading_date_ts = bdates[-1]
    num_trading_sessions = int(len(bdates))

    feature_cols = metadata["feature_columns"]
    scaler = metadata["scaler"]
    trained_models = metadata["trained_models"]
    model_bundle = trained_models[active_model_name]
    estimator = model_bundle["model"]
    use_scaled = model_bundle["use_scaled"]

    # Retrieve validation RMSE of the active model for uncertainty fan calculation
    metrics_df = metadata["metrics_df"]
    model_row = metrics_df[metrics_df["Model"] == active_model_name]
    base_rmse = float(model_row["RMSE"].iloc[0]) if not model_row.empty else float(metadata["best_model_metrics"]["RMSE"])

    # Initialize rolling price history with the last 30 actual closing prices
    price_history: List[float] = clean_df["Close"].tail(30).astype(float).tolist()

    forecast_records = []
    for step_idx, session_dt in enumerate(bdates, start=1):
        X_step_df = _build_step_feature_vector(price_history, feature_cols)
        X_in = scaler.transform(X_step_df) if use_scaled else X_step_df.values
        step_pred = float(estimator.predict(X_in)[0])

        # Append predicted price to rolling history for subsequent recursive steps
        price_history.append(step_pred)

        # Random-walk error propagation: sigma_h = RMSE_1 * sqrt(h)
        horizon_std = base_rmse * np.sqrt(step_idx)
        abs_diff_from_latest = step_pred - latest_price
        pct_diff_from_latest = (abs_diff_from_latest / latest_price) * 100.0 if latest_price != 0 else 0.0

        forecast_records.append({
            "Step": step_idx,
            "Trading_Date": session_dt,
            "Trading_Date_Str": session_dt.strftime("%Y-%m-%d"),
            "Predicted_Price": step_pred,
            "Abs_Change_From_Latest": abs_diff_from_latest,
            "Pct_Change_From_Latest": pct_diff_from_latest,
            "Uncertainty_Std": horizon_std,
            "Lower_68": max(0.0, step_pred - horizon_std),
            "Upper_68": step_pred + horizon_std,
            "Lower_95": max(0.0, step_pred - 1.96 * horizon_std),
            "Upper_95": step_pred + 1.96 * horizon_std,
        })

    forecast_df = pd.DataFrame(forecast_records)
    final_row = forecast_df.iloc[-1]

    predicted_price = float(final_row["Predicted_Price"])
    abs_change = float(final_row["Abs_Change_From_Latest"])
    pct_change = float(final_row["Pct_Change_From_Latest"])

    selected_date_str = target_ts.strftime("%Y-%m-%d")
    selected_day_name = target_ts.strftime("%A")
    effective_trading_date_str = effective_trading_date_ts.strftime("%Y-%m-%d")
    effective_day_name = effective_trading_date_ts.strftime("%A")

    uncertainty_info = _get_horizon_uncertainty_assessment(
        calendar_days=calendar_days,
        trading_sessions=num_trading_sessions,
        is_weekend_target=is_weekend_target,
        selected_date_str=selected_date_str,
        effective_trading_date_str=effective_trading_date_str,
    )

    return {
        "selected_date": selected_date_str,
        "selected_day_name": selected_day_name,
        "effective_trading_date": effective_trading_date_str,
        "effective_day_name": effective_day_name,
        "is_weekend_target": is_weekend_target,
        "latest_date": base_pipeline["latest_date"],
        "latest_day_name": base_pipeline["latest_day_name"],
        "latest_price": latest_price,
        "predicted_price": predicted_price,
        "absolute_change": abs_change,
        "percentage_change": pct_change,
        "calendar_days": calendar_days,
        "estimated_trading_sessions": num_trading_sessions,
        "active_model_name": active_model_name,
        "best_model_name": best_model_name,
        "uncertainty_level": uncertainty_info["uncertainty_level"],
        "badge_color": uncertainty_info["badge_color"],
        "warning_message": uncertainty_info["warning_message"],
        "lower_68": float(final_row["Lower_68"]),
        "upper_68": float(final_row["Upper_68"]),
        "lower_95": float(final_row["Lower_95"]),
        "upper_95": float(final_row["Upper_95"]),
        "forecast_df": forecast_df,
        "unit_info": base_pipeline["unit_info"],
        "clean_df": clean_df,
    }
