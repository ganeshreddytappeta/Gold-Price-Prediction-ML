"""
train_model.py
--------------
Trains and evaluates four Supervised Machine Learning Regression models:
1. Linear Regression
2. Decision Tree Regression
3. Random Forest Regression
4. Gradient Boosting Regression

Compares all four models against a Persistence Baseline (Next Price = Latest Available Price)
using chronological train-test splitting (no random shuffling) and strict data-leakage prevention.
Saves the best model and complete training metadata using Joblib.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import hashlib
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data_loader import load_and_inspect_dataset
from src.preprocessing import preprocess_gold_data
from src.feature_engineering import (
    create_features,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    FEATURE_DESCRIPTIONS,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_gold_model.joblib"
METADATA_PATH = MODELS_DIR / "training_metadata.joblib"


def compute_file_hash(file_path: str) -> str:
    """Compute SHA-256 hash of the CSV dataset file to detect dataset changes."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate standard regression evaluation metrics:
    - Mean Absolute Error (MAE)
    - Mean Squared Error (MSE)
    - Root Mean Squared Error (RMSE)
    - R-squared (R²)
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
    }


def train_and_evaluate_models(
    custom_data_path: Optional[str] = None,
    train_ratio: float = 0.80,
) -> Dict[str, Any]:
    """
    Execute the complete end-to-end training and evaluation pipeline:
    1. Load and inspect the dataset.
    2. Preprocess and clean records chronologically.
    3. Engineer leakage-free historical lag and moving-average features.
    4. Split chronologically into Training (80%) and Validation/Test (20%) sets.
    5. Fit StandardScaler strictly on Training data only.
    6. Train all 4 regression models and evaluate against the Persistence Baseline.
    7. Select the best model based on chronological validation RMSE / MAE / R².
    8. Save the best model and metadata to `models/` using Joblib.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load & inspect dataset
    raw_df, inspection_report = load_and_inspect_dataset(custom_data_path)
    date_col = inspection_report["date_column"]
    price_col = inspection_report["price_column"]

    # 2. Preprocess data
    clean_df, preprocessing_summary = preprocess_gold_data(raw_df, date_col, price_col)

    # 3. Feature engineering
    supervised_df, full_feature_df, latest_feature_row = create_features(clean_df)

    # 4. Chronological Train-Test Split (NEVER shuffle time-series observations)
    n_total = len(supervised_df)
    split_idx = int(n_total * train_ratio)
    if split_idx < 30 or (n_total - split_idx) < 10:
        raise ValueError(
            f"Insufficient rows ({n_total}) for an {int(train_ratio*100)}/{int((1-train_ratio)*100)} "
            "chronological train-test split."
        )

    train_df = supervised_df.iloc[:split_idx].copy()
    test_df = supervised_df.iloc[split_idx:].copy()

    X_train = train_df[FEATURE_COLUMNS].copy()
    y_train = train_df[TARGET_COLUMN].values

    X_test = test_df[FEATURE_COLUMNS].copy()
    y_test = test_df[TARGET_COLUMN].values

    # 5. Fit preprocessing transformation (StandardScaler) on TRAINING DATA ONLY
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 6. Define the 4 Supervised Regression Models
    models_config = {
        "Linear Regression": {
            "estimator": LinearRegression(),
            "use_scaled": True,
        },
        "Decision Tree Regression": {
            "estimator": DecisionTreeRegressor(
                max_depth=6,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
            ),
            "use_scaled": False,
        },
        "Random Forest Regression": {
            "estimator": RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=6,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1,
            ),
            "use_scaled": False,
        },
        "Gradient Boosting Regression": {
            "estimator": GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                min_samples_split=6,
                min_samples_leaf=3,
                random_state=42,
            ),
            "use_scaled": False,
        },
    }

    trained_models: Dict[str, Any] = {}
    metrics_records = []
    test_predictions: Dict[str, np.ndarray] = {}
    feature_importances: Dict[str, pd.DataFrame] = {}

    # Evaluate Persistence Baseline first: Next Price = Latest Available Price (Current_Close)
    baseline_train_pred = X_train["Current_Close"].values
    baseline_test_pred = X_test["Current_Close"].values
    baseline_train_metrics = evaluate_predictions(y_train, baseline_train_pred)
    baseline_test_metrics = evaluate_predictions(y_test, baseline_test_pred)

    test_predictions["Persistence Baseline"] = baseline_test_pred
    metrics_records.append({
        "Model": "Persistence Baseline (Naive)",
        "Type": "Baseline",
        "MAE": baseline_test_metrics["MAE"],
        "MSE": baseline_test_metrics["MSE"],
        "RMSE": baseline_test_metrics["RMSE"],
        "R2": baseline_test_metrics["R2"],
        "Train_MAE": baseline_train_metrics["MAE"],
        "Train_RMSE": baseline_train_metrics["RMSE"],
        "Train_R2": baseline_train_metrics["R2"],
    })

    # Train and evaluate each of the 4 ML models
    for model_name, cfg in models_config.items():
        estimator = cfg["estimator"]
        use_scaled = cfg["use_scaled"]

        X_tr = X_train_scaled if use_scaled else X_train.values
        X_te = X_test_scaled if use_scaled else X_test.values

        estimator.fit(X_tr, y_train)
        y_train_pred = estimator.predict(X_tr)
        y_test_pred = estimator.predict(X_te)

        train_m = evaluate_predictions(y_train, y_train_pred)
        test_m = evaluate_predictions(y_test, y_test_pred)

        trained_models[model_name] = {
            "model": estimator,
            "use_scaled": use_scaled,
        }
        test_predictions[model_name] = y_test_pred

        metrics_records.append({
            "Model": model_name,
            "Type": "Machine Learning",
            "MAE": test_m["MAE"],
            "MSE": test_m["MSE"],
            "RMSE": test_m["RMSE"],
            "R2": test_m["R2"],
            "Train_MAE": train_m["MAE"],
            "Train_RMSE": train_m["RMSE"],
            "Train_R2": train_m["R2"],
        })

        # Extract feature importance for tree-based models
        if hasattr(estimator, "feature_importances_"):
            imp_values = estimator.feature_importances_
            imp_df = pd.DataFrame({
                "Feature": FEATURE_COLUMNS,
                "Importance": imp_values,
                "Importance_Pct": imp_values * 100.0,
                "Description": [FEATURE_DESCRIPTIONS.get(f, "") for f in FEATURE_COLUMNS],
            }).sort_values("Importance", ascending=False).reset_index(drop=True)
            feature_importances[model_name] = imp_df

    metrics_df = pd.DataFrame(metrics_records)

    # Select best ML model based on chronological validation/test RMSE (lowest RMSE)
    ml_metrics_df = metrics_df[metrics_df["Type"] == "Machine Learning"].sort_values(
        by=["RMSE", "MAE"], ascending=[True, True]
    ).reset_index(drop=True)

    best_model_name = str(ml_metrics_df.iloc[0]["Model"])
    best_model_info = trained_models[best_model_name]
    best_model_metrics = ml_metrics_df.iloc[0].to_dict()

    # Build chronological test results DataFrame for plotting Actual vs. Predicted
    test_results_df = pd.DataFrame({
        "Date": test_df["Date"].values,
        "Target_Date": test_df["Target_Date"].values,
        "Current_Close": test_df["Current_Close"].values,
        "Actual_Next_Close": y_test,
        "Persistence Baseline": test_predictions["Persistence Baseline"],
    })
    for m_name in models_config.keys():
        test_results_df[m_name] = test_predictions[m_name]

    dataset_hash = compute_file_hash(inspection_report["file_path"])

    # Save best model bundle to `models/best_gold_model.joblib`
    best_model_bundle = {
        "model_name": best_model_name,
        "model": best_model_info["model"],
        "use_scaled": best_model_info["use_scaled"],
        "scaler": scaler,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "metrics": best_model_metrics,
        "unit_info": inspection_report["unit_info"],
        "dataset_hash": dataset_hash,
    }
    joblib.dump(best_model_bundle, BEST_MODEL_PATH)

    # Save comprehensive training metadata to `models/training_metadata.joblib`
    training_metadata = {
        "best_model_name": best_model_name,
        "best_model_metrics": best_model_metrics,
        "trained_models": trained_models,
        "scaler": scaler,
        "feature_columns": FEATURE_COLUMNS,
        "feature_descriptions": FEATURE_DESCRIPTIONS,
        "target_column": TARGET_COLUMN,
        "metrics_df": metrics_df,
        "feature_importances": feature_importances,
        "test_results_df": test_results_df,
        "split_info": {
            "total_supervised_rows": n_total,
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "train_ratio": train_ratio,
            "train_start_date": train_df["Date"].min().strftime("%Y-%m-%d"),
            "train_end_date": train_df["Date"].max().strftime("%Y-%m-%d"),
            "test_start_date": test_df["Date"].min().strftime("%Y-%m-%d"),
            "test_end_date": test_df["Date"].max().strftime("%Y-%m-%d"),
        },
        "inspection_report": inspection_report,
        "preprocessing_summary": preprocessing_summary,
        "dataset_hash": dataset_hash,
    }
    joblib.dump(training_metadata, METADATA_PATH)

    return training_metadata


if __name__ == "__main__":
    print("Starting Gold Price Prediction model training pipeline...")
    metadata = train_and_evaluate_models()
    print(f"\nBest Selected Model (Chronological Validation): {metadata['best_model_name']}")
    print("\nModel Comparison Table (Test Set):")
    print(metadata["metrics_df"][["Model", "Type", "MAE", "MSE", "RMSE", "R2"]].to_string(index=False))
    print(f"\nSaved best model to: {BEST_MODEL_PATH}")
    print(f"Saved training metadata to: {METADATA_PATH}")

