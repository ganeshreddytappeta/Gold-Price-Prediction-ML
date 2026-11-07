# 🪙 Gold Price Prediction Using Supervised Machine Learning

A complete, modular, intermediate-level **Gold Price Prediction Web Application** built with **Python**, **Scikit-Learn**, **Plotly**, and **Streamlit**.

---

## 📌 Project Overview

This project predicts the **next available trading session's gold price** using 10 years of historical trading data (`2016-01-29` to `2026-01-23`, 2,511 sessions) from Kaggle (`gold_prices_10y.csv` / `data/gold_data.csv`).

### Crucial Dataset & Price Unit Finding
- Upon inspecting the Kaggle dataset (`gold_prices_10y.csv`), the historical prices range from **$106.95** (Jan 2016) to **$458.00** (Jan 2026) with daily volumes in millions of shares and `OHLCV + Adj Close` columns.
- These values represent **US Dollars (USD) per Share of the SPDR Gold Shares ETF (`GLD`)** (where 1 share tracks approximately 1/10th of a troy ounce of physical gold bullion), **not** Indian Rupees (INR) per gram.
- Because the dataset does not contain historical `USD/INR` exchange rates, the pipeline strictly preserves the authentic dataset unit (**USD per Share — GLD Benchmark**) throughout model training, evaluation, and primary visualization without silently inventing exchange rates, while offering an optional, clearly labeled user-input currency converter in the sidebar.

---

## 📂 Project Folder Structure

```text
Gold-Price-Prediction/
│
├── data/
│   └── gold_data.csv              # Historical Kaggle Gold dataset (2,511 trading days)
│
├── models/
│   ├── best_gold_model.joblib     # Saved best regression model + scaler + unit metadata
│   └── training_metadata.joblib   # Saved comparison metrics, test predictions & feature importances
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py             # Automatic column detection, unit identification & dataset audit
│   ├── preprocessing.py           # Chronological sorting, cleaning & Daily/Weekly/Monthly resampling
│   ├── feature_engineering.py     # Leakage-free Lag (1, 2, 3, 7), MA (7, 14) & % change features
│   ├── train_model.py             # Chronological split, 4 ML regressors + Persistence Baseline
│   └── prediction.py              # Next-session & multi-step recursive custom future date forecasting
│
├── .streamlit/
│   └── config.toml                # Custom Gold & Dark Streamlit theme configuration
├── app.py                         # Interactive Streamlit Dashboard (Sections A–F + Custom Date Prediction)
├── requirements.txt               # Project dependencies
├── README.md                      # Project documentation
└── .gitignore                     # Git ignore rules
```

---

## 🚀 Setup & Execution Instructions

### 1. Create and Activate Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Train & Evaluate Models (Optional — Also Runs Automatically if Models Are Missing)
```powershell
python -m src.train_model
```

### 4. Launch the Streamlit Web Application
```powershell
streamlit run app.py
```

---

## 🧠 Machine Learning Pipeline & Data Leakage Prevention

1. **Chronological Train-Test Split (80% Train / 20% Validation):**
   - **Never shuffles time-series records.**
   - Training set covers `2016–2024`; Validation/Test set covers `2024–2026`.
2. **Causal Feature Engineering:**
   - `Current_Close` ($P_t$), `Lag_1` ($P_{t-1}$), `Lag_2` ($P_{t-2}$), `Lag_3` ($P_{t-3}$), `Lag_7` ($P_{t-7}$)
   - `MA_7` (7-session rolling mean), `MA_14` (14-session rolling mean)
   - `Pct_Change` (1-session % change), `Pct_Change_7` (7-session % change), `Volatility_7` (7-session rolling return std)
   - Target variable: `Target_Next_Close` ($P_{t+1}$, next available trading session's closing price).
3. **Preprocessing Isolation:**
   - `StandardScaler` is fit **strictly on `X_train`** and applied to `X_test`.
4. **Evaluated Algorithms:**
   - **Persistence Baseline** ($\hat{y}_{t+1} = P_t$)
   - **Linear Regression**
   - **Decision Tree Regression**
   - **Random Forest Regression**
   - **Gradient Boosting Regression**

