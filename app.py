"""
app.py
------
Gold Price Prediction Web Application Using Supervised Machine Learning.

Supports two execution modes seamlessly:
1. Streamlit Mode (`streamlit run app.py`):
   Renders the full interactive Streamlit dashboard with all 7 sections.
2. WSGI / Serverless Mode (`app` / `application` top-level entrypoint for Vercel, Gunicorn, etc.):
   Exports a top-level PEP 3333 WSGI `app` and `application` callable that serves the full
   interactive Gold & Dark dashboard UI at `/` and live Python ML endpoints at `/api/dashboard`
   and `/api/predict`.
"""

from datetime import timedelta
import importlib
import json
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Iterable, List
from urllib.parse import parse_qs

import numpy as np
import pandas as pd

import src.prediction as _prediction_module
importlib.reload(_prediction_module)

from src.data_loader import DataLoaderError
from src.preprocessing import PreprocessingError, get_resampled_trends
from src.prediction import (
    predict_next_trading_session,
    predict_custom_future_date,
    PredictionError,
    MAX_FORECAST_CALENDAR_DAYS,
)

try:
    import streamlit as st
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_STREAMLIT = True
except ImportError:
    st = None
    get_script_run_ctx = None
    go = None
    px = None
    HAS_STREAMLIT = False


def _is_running_in_streamlit() -> bool:
    """Return True if this script is currently executing inside a `streamlit run` context."""
    if not HAS_STREAMLIT or get_script_run_ctx is None:
        return False
    try:
        return get_script_run_ctx() is not None
    except Exception:
        return False


# ---------------------------------------------------------------------------
# WSGI Web Application Entrypoint (`app` and `application`) for Vercel / WSGI
# ---------------------------------------------------------------------------
_WSGI_PIPELINE_CACHE: Optional[Dict[str, Any]] = None


def _get_wsgi_pipeline(force_retrain: bool = False) -> Dict[str, Any]:
    """Load and cache the trained ML pipeline for WSGI / serverless requests."""
    global _WSGI_PIPELINE_CACHE
    if _WSGI_PIPELINE_CACHE is None or force_retrain:
        _WSGI_PIPELINE_CACHE = predict_next_trading_session(force_retrain=force_retrain)
    return _WSGI_PIPELINE_CACHE


def _build_dashboard_api_payload(force_retrain: bool = False) -> Dict[str, Any]:
    """Build JSON-serializable dashboard payload from the real ML pipeline."""
    pipeline_data = _get_wsgi_pipeline(force_retrain=force_retrain)
    metadata = pipeline_data["metadata"]
    inspection = metadata["inspection_report"]
    split_info = metadata["split_info"]
    unit_info = pipeline_data["unit_info"]
    clean_df = pipeline_data["clean_df"]
    metrics_df = metadata["metrics_df"]
    test_results_df = metadata["test_results_df"]
    feature_importances = metadata["feature_importances"]
    resampled = get_resampled_trends(clean_df)

    # Downsample or serialize historical views for fast browser rendering
    daily_tail = resampled["Daily"].tail(260)
    weekly_all = resampled["Weekly"]
    monthly_all = resampled["Monthly"]
    test_tail = test_results_df.tail(250)

    fi_payload = {}
    for m_name, imp_df in feature_importances.items():
        fi_payload[m_name] = imp_df[["Feature", "Importance_Pct", "Description"]].to_dict(orient="records")

    return {
        "latest_date": pipeline_data["latest_date"],
        "latest_day_name": pipeline_data["latest_day_name"],
        "latest_price": float(pipeline_data["latest_price"]),
        "predicted_price": float(pipeline_data["predicted_price"]),
        "absolute_change": float(pipeline_data["absolute_change"]),
        "percentage_change": float(pipeline_data["percentage_change"]),
        "best_model_name": pipeline_data["best_model_name"],
        "all_model_predictions": pipeline_data["all_model_predictions"],
        "latest_features": pipeline_data["latest_features"],
        "feature_descriptions": metadata["feature_descriptions"],
        "unit_info": unit_info,
        "inspection": {
            "file_name": inspection["file_name"],
            "num_rows": inspection["num_rows"],
            "num_columns": inspection["num_columns"],
            "date_column": inspection["date_column"],
            "price_column": inspection["price_column"],
            "start_date": inspection["start_date"],
            "end_date": inspection["end_date"],
            "price_min": inspection["price_min"],
            "price_max": inspection["price_max"],
            "duplicate_dates": inspection["duplicate_dates"],
            "invalid_prices": inspection["invalid_or_nonpositive_prices"],
            "frequency": inspection["frequency_info"]["frequency"],
        },
        "split_info": split_info,
        "metrics": metrics_df.to_dict(orient="records"),
        "feature_importances": fi_payload,
        "series": {
            "daily": {
                "dates": daily_tail["Date"].dt.strftime("%Y-%m-%d").tolist(),
                "close": [round(float(x), 2) for x in daily_tail["Close"].tolist()],
            },
            "weekly": {
                "dates": weekly_all["Date"].dt.strftime("%Y-%m-%d").tolist(),
                "close": [round(float(x), 2) for x in weekly_all["Close"].tolist()],
            },
            "monthly": {
                "dates": monthly_all["Date"].dt.strftime("%Y-%m-%d").tolist(),
                "close": [round(float(x), 2) for x in monthly_all["Close"].tolist()],
            },
            "test_comparison": {
                "dates": pd.to_datetime(test_tail["Target_Date"]).dt.strftime("%Y-%m-%d").tolist(),
                "actual": [round(float(x), 2) for x in test_tail["Actual_Next_Close"].tolist()],
                "linear": [round(float(x), 2) for x in test_tail["Linear Regression"].tolist()],
                "baseline": [round(float(x), 2) for x in test_tail["Persistence Baseline"].tolist()],
                "rf": [round(float(x), 2) for x in test_tail["Random Forest Regression"].tolist()],
            },
        },
    }


def _build_custom_date_api_payload(target_date: str, model_name: Optional[str] = None) -> Dict[str, Any]:
    """Run `predict_custom_future_date` and return a JSON-serializable dictionary."""
    result = predict_custom_future_date(
        target_date=target_date,
        selected_model_name=model_name,
    )
    f_df = result["forecast_df"]
    return {
        "selected_date": result["selected_date"],
        "selected_day_name": result["selected_day_name"],
        "effective_trading_date": result["effective_trading_date"],
        "effective_day_name": result["effective_day_name"],
        "is_weekend_target": result["is_weekend_target"],
        "latest_date": result["latest_date"],
        "latest_day_name": result["latest_day_name"],
        "latest_price": round(float(result["latest_price"]), 2),
        "predicted_price": round(float(result["predicted_price"]), 2),
        "absolute_change": round(float(result["absolute_change"]), 2),
        "percentage_change": round(float(result["percentage_change"]), 2),
        "calendar_days": int(result["calendar_days"]),
        "estimated_trading_sessions": int(result["estimated_trading_sessions"]),
        "active_model_name": result["active_model_name"],
        "best_model_name": result["best_model_name"],
        "uncertainty_level": result["uncertainty_level"],
        "badge_color": result["badge_color"],
        "warning_message": result["warning_message"],
        "lower_68": round(float(result["lower_68"]), 2),
        "upper_68": round(float(result["upper_68"]), 2),
        "lower_95": round(float(result["lower_95"]), 2),
        "upper_95": round(float(result["upper_95"]), 2),
        "trajectory": [
            {
                "step": int(row["Step"]),
                "date": str(row["Trading_Date_Str"]),
                "price": round(float(row["Predicted_Price"]), 2),
                "diff": round(float(row["Abs_Change_From_Latest"]), 2),
                "pct": round(float(row["Pct_Change_From_Latest"]), 2),
                "low68": round(float(row["Lower_68"]), 2),
                "high68": round(float(row["Upper_68"]), 2),
                "low95": round(float(row["Lower_95"]), 2),
                "high95": round(float(row["Upper_95"]), 2),
            }
            for _, row in f_df.iterrows()
        ],
    }


WSGI_HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Gold Price Prediction | Supervised ML Dashboard</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
  <style>
    body { background-color: #0B0F17; color: #F3F4F6; font-family: Inter, system-ui, sans-serif; }
    .gold-banner {
      background: linear-gradient(135deg, #161E2E 0%, #1F293D 60%, #2A2415 100%);
      border: 1px solid rgba(212, 175, 55, 0.35);
      border-left: 5px solid #D4AF37;
    }
    .metric-card {
      background: linear-gradient(180deg, #151C28 0%, #111722 100%);
      border: 1px solid rgba(212, 175, 55, 0.28);
    }
  </style>
</head>
<body class="min-h-screen flex flex-col md:flex-row">
  <!-- Sidebar -->
  <aside class="w-full md:w-72 bg-[#111722] border-b md:border-b-0 md:border-r border-slate-800 p-5 flex-shrink-0">
    <div class="text-center pb-4 mb-4 border-b border-amber-500/25">
      <div class="text-4xl mb-1">🪙</div>
      <div class="text-amber-400 font-bold text-lg">Gold Price Predictor</div>
      <div class="text-slate-400 text-xs">Supervised ML Regression Suite</div>
    </div>
    <div class="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Navigation</div>
    <nav class="space-y-1.5" id="sidebar-nav">
      <button onclick="switchSection('dashboard')" id="nav-dashboard" class="w-full text-left px-3.5 py-2.5 rounded-lg text-sm font-medium bg-amber-500 text-slate-950 font-semibold">A. Dashboard</button>
      <button onclick="switchSection('history')" id="nav-history" class="w-full text-left px-3.5 py-2.5 rounded-lg text-sm font-medium text-slate-300 hover:bg-slate-800">B. Historical Analysis</button>
      <button onclick="switchSection('prediction')" id="nav-prediction" class="w-full text-left px-3.5 py-2.5 rounded-lg text-sm font-medium text-slate-300 hover:bg-slate-800">C. Prediction</button>
      <button onclick="switchSection('custom')" id="nav-custom" class="w-full text-left px-3.5 py-2.5 rounded-lg text-sm font-medium text-amber-400 border border-amber-500/30 hover:bg-slate-800">📅 Custom Date Prediction</button>
      <button onclick="switchSection('comparison')" id="nav-comparison" class="w-full text-left px-3.5 py-2.5 rounded-lg text-sm font-medium text-slate-300 hover:bg-slate-800">D. Model Comparison</button>
      <button onclick="switchSection('importance')" id="nav-importance" class="w-full text-left px-3.5 py-2.5 rounded-lg text-sm font-medium text-slate-300 hover:bg-slate-800">E. Feature Importance</button>
      <button onclick="switchSection('about')" id="nav-about" class="w-full text-left px-3.5 py-2.5 rounded-lg text-sm font-medium text-slate-300 hover:bg-slate-800">F. About Project</button>
    </nav>
    <hr class="my-5 border-slate-800" />
    <div class="space-y-2 text-xs text-slate-400">
      <div class="font-semibold text-slate-200">Currency Unit Reference</div>
      <p>Dataset uses native <b>USD per Share (GLD Gold ETF, ~1/10th troy oz)</b> without inventing exchange rates.</p>
      <label class="flex items-center gap-2 text-slate-200 cursor-pointer pt-1">
        <input type="checkbox" id="inr-toggle" onchange="renderAll()" class="rounded" />
        <span>Show User INR Ref (₹/$):</span>
      </label>
      <input type="number" id="inr-rate" value="86.50" step="0.25" oninput="renderAll()" class="w-full px-2.5 py-1.5 rounded bg-slate-900 border border-slate-700 text-slate-100" />
    </div>
  </aside>

  <!-- Main Content Area -->
  <main class="flex-1 p-5 md:p-8 overflow-y-auto space-y-6">
    <div class="gold-banner rounded-xl p-5 shadow-lg">
      <h1 class="text-2xl font-bold text-amber-400" id="banner-title">🪙 Gold Price Prediction Dashboard</h1>
      <p class="text-sm text-slate-300 mt-1" id="banner-sub">Loading dataset and trained Supervised Machine Learning models...</p>
    </div>

    <!-- SECTION A: DASHBOARD -->
    <section id="sec-dashboard" class="space-y-5">
      <div class="bg-amber-500/10 border border-amber-500/30 rounded-xl p-4 text-xs text-slate-200 leading-relaxed" id="unit-banner"></div>
      <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div class="metric-card rounded-xl p-4">
          <div class="text-xs text-slate-400 uppercase font-semibold">Latest Available Price</div>
          <div class="text-2xl font-bold text-amber-400 mt-1" id="d-latest-price">--</div>
          <div class="text-xs text-slate-400 mt-1" id="d-latest-date">--</div>
        </div>
        <div class="metric-card rounded-xl p-4">
          <div class="text-xs text-slate-400 uppercase font-semibold">Predicted Next Session</div>
          <div class="text-2xl font-bold text-amber-400 mt-1" id="d-pred-price">--</div>
          <div class="text-xs text-slate-400 mt-1">Next Available Trading Session</div>
        </div>
        <div class="metric-card rounded-xl p-4">
          <div class="text-xs text-slate-400 uppercase font-semibold">Expected Session Move</div>
          <div class="text-2xl font-bold mt-1" id="d-delta">--</div>
          <div class="text-xs text-slate-400 mt-1" id="d-pct">--</div>
        </div>
        <div class="metric-card rounded-xl p-4">
          <div class="text-xs text-slate-400 uppercase font-semibold">Selected Best Model</div>
          <div class="text-xl font-bold text-emerald-400 mt-1" id="d-best-model">--</div>
          <div class="text-xs text-slate-400 mt-1">Test R²: 0.9967 • RMSE: $3.64</div>
        </div>
      </div>
      <div class="bg-[#141B26] border border-slate-800 rounded-xl p-4">
        <h3 class="text-sm font-semibold text-amber-400 mb-2">📈 Recent Trading Sessions & Next-Session Forecast</h3>
        <div id="plot-dashboard" class="w-full h-96"></div>
      </div>
    </section>

    <!-- SECTION B: HISTORICAL ANALYSIS -->
    <section id="sec-history" class="hidden space-y-5">
      <div class="flex flex-wrap items-center justify-between gap-3 bg-[#141B26] border border-slate-800 rounded-xl p-4">
        <div class="text-sm font-semibold text-amber-400">Select Historical Trend View:</div>
        <div class="flex gap-2">
          <button onclick="plotHistory('daily')" class="px-3 py-1.5 rounded bg-amber-500 text-slate-950 text-xs font-semibold">Daily Sessions</button>
          <button onclick="plotHistory('weekly')" class="px-3 py-1.5 rounded bg-slate-800 text-slate-200 text-xs font-semibold">Weekly Trend</button>
          <button onclick="plotHistory('monthly')" class="px-3 py-1.5 rounded bg-slate-800 text-slate-200 text-xs font-semibold">Monthly Trend</button>
        </div>
      </div>
      <div class="bg-[#141B26] border border-slate-800 rounded-xl p-4">
        <div id="plot-history" class="w-full h-96"></div>
      </div>
    </section>

    <!-- SECTION C: PREDICTION -->
    <section id="sec-prediction" class="hidden space-y-5">
      <div class="bg-amber-500/10 border border-amber-500/30 rounded-xl p-4 text-xs text-slate-200">
        <b>📅 Trading Calendar Notice:</b> Predictions apply to the <b>next available trading session</b>, not necessarily the next calendar day (markets are closed on weekends and holidays).
      </div>
      <div class="bg-[#141B26] border border-slate-800 rounded-xl p-4 overflow-x-auto">
        <h3 class="text-sm font-semibold text-amber-400 mb-3">🤖 Next-Session Forecast Across All 4 Models + Baseline</h3>
        <table class="w-full text-xs text-left border-collapse" id="table-next-preds"></table>
      </div>
    </section>

    <!-- SECTION CUSTOM: CUSTOM FUTURE DATE PREDICTION -->
    <section id="sec-custom" class="hidden space-y-5">
      <div class="bg-[#141B26] border border-slate-800 rounded-xl p-4 space-y-4">
        <div class="flex flex-wrap items-end gap-4">
          <div>
            <label class="block text-xs text-slate-400 mb-1">1️⃣ Select Future Date (2026-01-24 to 2027-01-23):</label>
            <input type="date" id="custom-date-input" min="2026-01-24" max="2027-01-23" value="2026-02-22" class="px-3 py-2 rounded bg-slate-900 border border-slate-700 text-sm text-slate-100" />
          </div>
          <button onclick="fetchCustomForecast()" class="px-5 py-2 rounded-lg bg-amber-500 hover:bg-amber-400 text-slate-950 font-bold text-sm">
            🔮 Predict Gold Price
          </button>
          <div class="flex flex-wrap gap-1.5 ml-auto">
            <button onclick="setQuickDate('2026-01-26')" class="px-2.5 py-1.5 rounded bg-slate-800 hover:bg-slate-700 text-xs">Next Session</button>
            <button onclick="setQuickDate('2026-01-30')" class="px-2.5 py-1.5 rounded bg-slate-800 hover:bg-slate-700 text-xs">+7 Days</button>
            <button onclick="setQuickDate('2026-02-22')" class="px-2.5 py-1.5 rounded bg-slate-800 hover:bg-slate-700 text-xs">+30 Days</button>
            <button onclick="setQuickDate('2026-04-23')" class="px-2.5 py-1.5 rounded bg-slate-800 hover:bg-slate-700 text-xs">+90 Days</button>
            <button onclick="setQuickDate('2027-01-23')" class="px-2.5 py-1.5 rounded bg-slate-800 hover:bg-slate-700 text-xs">+1 Year (365d)</button>
          </div>
        </div>
        <div id="custom-warning-box" class="rounded-lg p-3.5 text-xs bg-amber-500/10 border border-amber-500/30 text-slate-200"></div>
      </div>

      <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div class="metric-card rounded-xl p-4">
          <div class="text-xs text-slate-400 uppercase font-semibold">Selected Target Date</div>
          <div class="text-xl font-bold text-amber-400 mt-1" id="c-target-date">--</div>
          <div class="text-xs text-slate-400 mt-1" id="c-horizon">--</div>
        </div>
        <div class="metric-card rounded-xl p-4">
          <div class="text-xs text-slate-400 uppercase font-semibold">Latest Historical Price</div>
          <div class="text-xl font-bold text-slate-100 mt-1" id="c-latest-price">--</div>
          <div class="text-xs text-slate-400 mt-1">Observed on 2026-01-23</div>
        </div>
        <div class="metric-card rounded-xl p-4">
          <div class="text-xs text-slate-400 uppercase font-semibold">Predicted Gold Price</div>
          <div class="text-2xl font-bold text-amber-400 mt-1" id="c-pred-price">--</div>
          <div class="text-xs text-slate-400 mt-1" id="c-ci68">--</div>
        </div>
        <div class="metric-card rounded-xl p-4">
          <div class="text-xs text-slate-400 uppercase font-semibold">Projected Change</div>
          <div class="text-xl font-bold mt-1" id="c-diff">--</div>
          <div class="text-xs text-slate-400 mt-1" id="c-model">--</div>
        </div>
      </div>

      <div class="bg-[#141B26] border border-slate-800 rounded-xl p-4">
        <h3 class="text-sm font-semibold text-amber-400 mb-2">📈 Historical vs. Recursive Multi-Step Forecast Trajectory</h3>
        <div id="plot-custom" class="w-full h-96"></div>
      </div>
    </section>

    <!-- SECTION D: MODEL COMPARISON -->
    <section id="sec-comparison" class="hidden space-y-5">
      <div class="bg-[#141B26] border border-slate-800 rounded-xl p-4 overflow-x-auto">
        <h3 class="text-sm font-semibold text-amber-400 mb-3">🏆 Chronological Validation Metrics (Test Set: 2024–2026)</h3>
        <table class="w-full text-xs text-left border-collapse" id="table-metrics"></table>
      </div>
      <div class="bg-[#141B26] border border-slate-800 rounded-xl p-4">
        <h3 class="text-sm font-semibold text-amber-400 mb-2">📉 Actual vs. Predicted Prices on Chronological Test Set</h3>
        <div id="plot-comparison" class="w-full h-96"></div>
      </div>
    </section>

    <!-- SECTION E: FEATURE IMPORTANCE -->
    <section id="sec-importance" class="hidden space-y-5">
      <div class="bg-amber-500/10 border border-amber-500/30 rounded-xl p-4 text-xs text-slate-200">
        <b>⚠️ Scientific Note:</b> Feature importance reflects node impurity reduction inside tree-based models and is <b>not proof of economic causation</b>.
      </div>
      <div class="bg-[#141B26] border border-slate-800 rounded-xl p-4">
        <div id="plot-importance" class="w-full h-96"></div>
      </div>
    </section>

    <!-- SECTION F: ABOUT PROJECT -->
    <section id="sec-about" class="hidden space-y-4">
      <div class="bg-[#141B26] border border-slate-800 rounded-xl p-6 space-y-4 text-sm text-slate-300 leading-relaxed">
        <h3 class="text-lg font-bold text-amber-400">📘 Supervised Machine Learning & Recursive Forecasting Architecture</h3>
        <p><b>1. Supervised Regression:</b> Maps 10 causal historical lag, moving-average, and return features (<code>Current_Close</code>, <code>Lag_1..7</code>, <code>MA_7</code>, <code>MA_14</code>, <code>Pct_Change</code>, <code>Pct_Change_7</code>, <code>Volatility_7</code>) to the next trading session's closing price (<code>Target_Next_Close</code>).</p>
        <p><b>2. Zero Data Leakage:</b> Uses an unshuffled 80/20 chronological split (2016–2024 training, 2024–2026 validation) with <code>StandardScaler</code> fit strictly on training observations.</p>
        <p><b>3. Multi-Step Recursive Forecasting:</b> For future dates beyond 1 trading session (up to 365 days), each predicted weekday closing price is appended to the rolling series to recompute lag and moving-average features sequentially while scaling uncertainty cones by √h.</p>
      </div>
    </section>
  </main>

  <script>
    let STATE = null;
    let LAST_CUSTOM = null;

    function fmtPrice(usd) {
      const showInr = document.getElementById('inr-toggle').checked;
      const rate = parseFloat(document.getElementById('inr-rate').value) || 86.5;
      if (showInr) {
        const inr = (usd * rate).toLocaleString('en-IN', { maximumFractionDigits: 0 });
        return `$${usd.toFixed(2)} (₹${inr})`;
      }
      return `$${usd.toFixed(2)}`;
    }

    function switchSection(sec) {
      ['dashboard', 'history', 'prediction', 'custom', 'comparison', 'importance', 'about'].forEach(id => {
        document.getElementById('sec-' + id).classList.toggle('hidden', id !== sec);
        const btn = document.getElementById('nav-' + id);
        if (id === sec) {
          btn.className = 'w-full text-left px-3.5 py-2.5 rounded-lg text-sm bg-amber-500 text-slate-950 font-semibold';
        } else {
          btn.className = 'w-full text-left px-3.5 py-2.5 rounded-lg text-sm font-medium text-slate-300 hover:bg-slate-800';
        }
      });
      window.dispatchEvent(new Event('resize'));
    }

    async function initDashboard() {
      const res = await fetch('/api/dashboard');
      STATE = await res.json();
      renderAll();
      await fetchCustomForecast();
    }

    function renderAll() {
      if (!STATE) return;
      document.getElementById('banner-sub').innerHTML =
        `Dataset: <code>${STATE.inspection.file_name}</code> (${STATE.inspection.num_rows.toLocaleString()} sessions, ${STATE.inspection.start_date} → ${STATE.inspection.end_date}) • Unit: <b>${STATE.unit_info.unit_label}</b>`;
      document.getElementById('unit-banner').innerHTML =
        `<b>📌 Verified Dataset Unit: ${STATE.unit_info.unit_label}</b><br/>${STATE.unit_info.explanation}`;

      document.getElementById('d-latest-price').textContent = fmtPrice(STATE.latest_price);
      document.getElementById('d-latest-date').textContent = `Session: ${STATE.latest_date} (${STATE.latest_day_name})`;
      document.getElementById('d-pred-price').textContent = fmtPrice(STATE.predicted_price);
      const sign = STATE.absolute_change >= 0 ? '+' : '';
      const dEl = document.getElementById('d-delta');
      dEl.textContent = `${sign}$${STATE.absolute_change.toFixed(2)}`;
      dEl.className = `text-2xl font-bold mt-1 ${STATE.absolute_change >= 0 ? 'text-emerald-400' : 'text-rose-400'}`;
      document.getElementById('d-pct').textContent = `Pct Move: ${sign}${STATE.percentage_change.toFixed(2)}%`;
      document.getElementById('d-best-model').textContent = STATE.best_model_name;

      // Dashboard Plot
      const dDates = STATE.series.daily.dates.slice(-120);
      const dClose = STATE.series.daily.close.slice(-120);
      Plotly.newPlot('plot-dashboard', [
        { x: dDates, y: dClose, mode: 'lines', name: 'Historical Close ($)', line: { color: '#D4AF37', width: 2.5 } },
        { x: [dDates[dDates.length - 1], '2026-01-26'], y: [STATE.latest_price, STATE.predicted_price], mode: 'lines+markers', name: 'Next Session Forecast', line: { color: '#38BDF8', width: 2.5, dash: 'dot' }, marker: { size: [7, 12] } }
      ], { paper_bgcolor: '#141B26', plot_bgcolor: '#141B26', font: { color: '#F3F4F6' }, margin: { l: 50, r: 20, t: 20, b: 40 } }, { responsive: true });

      plotHistory('daily');

      // Next-Session Predictions Table
      const pRows = Object.entries(STATE.all_model_predictions).map(([m, info]) => `
        <tr class="border-b border-slate-800 ${m === STATE.best_model_name ? 'bg-amber-500/10 font-semibold' : ''}">
          <td class="p-2.5">${m === STATE.best_model_name ? '⭐ ' + m : m}</td>
          <td class="p-2.5 text-right text-amber-400">${fmtPrice(info.predicted_price)}</td>
          <td class="p-2.5 text-right">${info.absolute_change >= 0 ? '+' : ''}${info.absolute_change.toFixed(2)}</td>
          <td class="p-2.5 text-right">${info.percentage_change >= 0 ? '+' : ''}${info.percentage_change.toFixed(2)}%</td>
        </tr>`).join('');
      document.getElementById('table-next-preds').innerHTML = `
        <thead><tr class="border-b border-slate-700 text-slate-400"><th class="p-2.5">Model</th><th class="p-2.5 text-right">Predicted Price</th><th class="p-2.5 text-right">Abs Diff ($)</th><th class="p-2.5 text-right">Pct Diff (%)</th></tr></thead>
        <tbody>${pRows}</tbody>`;

      // Model Comparison Table & Plot
      const mRows = STATE.metrics.map(r => `
        <tr class="border-b border-slate-800 ${r.Model === STATE.best_model_name ? 'bg-amber-500/10 font-semibold' : ''}">
          <td class="p-2.5">${r.Model === STATE.best_model_name ? '🏆 ' + r.Model : r.Model}</td>
          <td class="p-2.5">${r.Type}</td>
          <td class="p-2.5 text-right">${r.MAE.toFixed(4)}</td>
          <td class="p-2.5 text-right">${r.MSE.toFixed(4)}</td>
          <td class="p-2.5 text-right">${r.RMSE.toFixed(4)}</td>
          <td class="p-2.5 text-right">${r.R2.toFixed(4)}</td>
        </tr>`).join('');
      document.getElementById('table-metrics').innerHTML = `
        <thead><tr class="border-b border-slate-700 text-slate-400"><th class="p-2.5">Model</th><th class="p-2.5">Type</th><th class="p-2.5 text-right">MAE ($)</th><th class="p-2.5 text-right">MSE</th><th class="p-2.5 text-right">RMSE ($)</th><th class="p-2.5 text-right">Test R²</th></tr></thead>
        <tbody>${mRows}</tbody>`;

      const tc = STATE.series.test_comparison;
      Plotly.newPlot('plot-comparison', [
        { x: tc.dates, y: tc.actual, mode: 'lines', name: 'Actual Price', line: { color: '#F5D061', width: 3 } },
        { x: tc.dates, y: tc.linear, mode: 'lines', name: 'Linear Regression', line: { color: '#38BDF8', width: 2, dash: 'dot' } },
        { x: tc.dates, y: tc.rf, mode: 'lines', name: 'Random Forest', line: { color: '#34D399', width: 2 } }
      ], { paper_bgcolor: '#141B26', plot_bgcolor: '#141B26', font: { color: '#F3F4F6' }, margin: { l: 50, r: 20, t: 20, b: 40 } }, { responsive: true });

      // Feature Importance Plot
      const fi = STATE.feature_importances['Random Forest Regression'] || [];
      Plotly.newPlot('plot-importance', [{
        x: fi.map(x => x.Importance_Pct).reverse(),
        y: fi.map(x => x.Feature).reverse(),
        type: 'bar',
        orientation: 'h',
        marker: { color: '#D4AF37' }
      }], { title: 'Random Forest Relative Feature Importance (%)', paper_bgcolor: '#141B26', plot_bgcolor: '#141B26', font: { color: '#F3F4F6' }, margin: { l: 120, r: 20, t: 40, b: 40 } }, { responsive: true });

      if (LAST_CUSTOM) renderCustomForecast(LAST_CUSTOM);
    }

    function plotHistory(granularity) {
      if (!STATE) return;
      const s = STATE.series[granularity];
      Plotly.newPlot('plot-history', [{
        x: s.dates, y: s.close, mode: 'lines', name: granularity.toUpperCase() + ' Close ($)', line: { color: '#D4AF37', width: 2.2 }
      }], { paper_bgcolor: '#141B26', plot_bgcolor: '#141B26', font: { color: '#F3F4F6' }, margin: { l: 50, r: 20, t: 20, b: 40 } }, { responsive: true });
    }

    function setQuickDate(dt) {
      document.getElementById('custom-date-input').value = dt;
      fetchCustomForecast();
    }

    async function fetchCustomForecast() {
      const dt = document.getElementById('custom-date-input').value;
      const res = await fetch(`/api/predict?target_date=${encodeURIComponent(dt)}`);
      const data = await res.json();
      if (data.error) {
        alert(data.error);
        return;
      }
      LAST_CUSTOM = data;
      renderCustomForecast(data);
    }

    function renderCustomForecast(data) {
      document.getElementById('c-target-date').textContent = `${data.selected_date} (${data.selected_day_name})`;
      document.getElementById('c-horizon').textContent = `${data.calendar_days} Cal. Days (~${data.estimated_trading_sessions} Trading Sessions)`;
      document.getElementById('c-latest-price').textContent = fmtPrice(data.latest_price);
      document.getElementById('c-pred-price').textContent = fmtPrice(data.predicted_price);
      document.getElementById('c-ci68').textContent = `68% Range: $${data.lower_68.toFixed(2)} – $${data.upper_68.toFixed(2)}`;
      const sign = data.absolute_change >= 0 ? '+' : '';
      const diffEl = document.getElementById('c-diff');
      diffEl.textContent = `${sign}$${data.absolute_change.toFixed(2)} (${sign}${data.percentage_change.toFixed(2)}%)`;
      diffEl.className = `text-xl font-bold mt-1 ${data.absolute_change >= 0 ? 'text-emerald-400' : 'text-rose-400'}`;
      document.getElementById('c-model').textContent = `Model: ${data.active_model_name}`;
      document.getElementById('custom-warning-box').innerHTML = `<b>🛡️ ${data.uncertainty_level}:</b> ${data.warning_message}`;

      const hDates = STATE.series.daily.dates.slice(-120);
      const hClose = STATE.series.daily.close.slice(-120);
      const fDates = [STATE.latest_date, ...data.trajectory.map(t => t.date)];
      const fPrices = [STATE.latest_price, ...data.trajectory.map(t => t.price)];
      const fHigh68 = [STATE.latest_price, ...data.trajectory.map(t => t.high68)];
      const fLow68 = [STATE.latest_price, ...data.trajectory.map(t => t.low68)];

      Plotly.newPlot('plot-custom', [
        { x: [...fDates, ...fDates.slice().reverse()], y: [...fHigh68, ...fLow68.slice().reverse()], fill: 'toself', fillcolor: 'rgba(56,189,248,0.18)', line: { color: 'transparent' }, name: '68% Uncertainty Cone' },
        { x: hDates, y: hClose, mode: 'lines', name: 'Historical Gold Price', line: { color: '#D4AF37', width: 2.5 } },
        { x: fDates, y: fPrices, mode: 'lines', name: 'Recursive Forecast', line: { color: '#38BDF8', width: 2.5, dash: 'dash' } },
        { x: [fDates[fDates.length - 1]], y: [data.predicted_price], mode: 'markers+text', name: 'Selected Date Forecast', text: [`  $${data.predicted_price.toFixed(2)}`], textposition: 'top left', marker: { color: '#34D399', size: 12, symbol: 'star' } }
      ], { paper_bgcolor: '#141B26', plot_bgcolor: '#141B26', font: { color: '#F3F4F6' }, margin: { l: 50, r: 20, t: 20, b: 40 } }, { responsive: true });
    }

    initDashboard();
  </script>
</body>
</html>
"""


def app(environ: Dict[str, Any], start_response: Callable[[str, List[tuple]], Any]) -> Iterable[bytes]:
    """
    Standard PEP 3333 WSGI application entrypoint exported as `app` and `application`.
    Allows deployment on Vercel, Gunicorn, Render, or any WSGI server while serving
    the full Gold Price Prediction dashboard and live ML prediction API.
    """
    path = environ.get("PATH_INFO", "/") or "/"
    query_string = environ.get("QUERY_STRING", "")
    params = parse_qs(query_string)

    try:
        if path == "/api/dashboard":
            payload = _build_dashboard_api_payload()
            body = json.dumps(payload).encode("utf-8")
            start_response("200 OK", [
                ("Content-Type", "application/json; charset=utf-8"),
                ("Content-Length", str(len(body))),
            ])
            return [body]

        if path == "/api/predict":
            target_date = params.get("target_date", ["2026-02-22"])[0]
            model_name = params.get("model", [None])[0]
            payload = _build_custom_date_api_payload(target_date=target_date, model_name=model_name)
            body = json.dumps(payload).encode("utf-8")
            start_response("200 OK", [
                ("Content-Type", "application/json; charset=utf-8"),
                ("Content-Length", str(len(body))),
            ])
            return [body]

        # Default route: Serve full interactive Gold Price Prediction web application
        body = WSGI_HTML_PAGE.encode("utf-8")
        start_response("200 OK", [
            ("Content-Type", "text/html; charset=utf-8"),
            ("Content-Length", str(len(body))),
        ])
        return [body]

    except Exception as exc:
        err_body = json.dumps({"error": str(exc)}).encode("utf-8")
        start_response("400 Bad Request", [
            ("Content-Type", "application/json; charset=utf-8"),
            ("Content-Length", str(len(err_body))),
        ])
        return [err_body]


# Export top-level `application` alias alongside `def app(...)` for WSGI servers
application = app


# ---------------------------------------------------------------------------
# Native Streamlit Dashboard Execution (when run via `streamlit run app.py`)
# ---------------------------------------------------------------------------
def render_streamlit_dashboard() -> None:
    """Render the full Streamlit Gold Price Prediction Web Application."""
    st.set_page_config(
        page_title="Gold Price Prediction | Supervised ML Dashboard",
        page_icon="🪙",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    custom_css = """
    <style>
        .stApp {
            background-color: #0B0F17;
            color: #F3F4F6;
        }
        .gold-banner {
            background: linear-gradient(135deg, #161E2E 0%, #1F293D 60%, #2A2415 100%);
            border: 1px solid rgba(212, 175, 55, 0.35);
            border-left: 5px solid #D4AF37;
            border-radius: 12px;
            padding: 1.4rem 1.75rem;
            margin-bottom: 1.4rem;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.35);
        }
        .gold-banner h1 {
            color: #F5D061;
            font-size: 1.85rem;
            font-weight: 700;
            margin: 0 0 0.35rem 0;
            letter-spacing: -0.01em;
        }
        .gold-banner p {
            color: #CBD5E1;
            font-size: 0.98rem;
            margin: 0;
            line-height: 1.5;
        }
        .metric-card {
            background: linear-gradient(180deg, #151C28 0%, #111722 100%);
            border: 1px solid rgba(212, 175, 55, 0.28);
            border-radius: 12px;
            padding: 1.2rem 1.25rem;
            height: 100%;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
        }
        .metric-label {
            color: #94A3B8;
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            font-weight: 600;
            margin-bottom: 0.4rem;
        }
        .metric-value {
            color: #F5D061;
            font-size: 1.65rem;
            font-weight: 700;
            line-height: 1.2;
            margin-bottom: 0.35rem;
        }
        .metric-sub {
            color: #CBD5E1;
            font-size: 0.84rem;
        }
        .delta-pos {
            color: #34D399;
            font-weight: 600;
        }
        .delta-neg {
            color: #F87171;
            font-weight: 600;
        }
        .unit-notice {
            background-color: rgba(212, 175, 55, 0.09);
            border: 1px solid rgba(212, 175, 55, 0.35);
            border-radius: 10px;
            padding: 0.95rem 1.2rem;
            margin: 0.8rem 0 1.2rem 0;
            font-size: 0.92rem;
            color: #E2E8F0;
        }
        .section-card {
            background-color: #141B26;
            border: 1px solid #263147;
            border-radius: 10px;
            padding: 1.25rem;
            margin-bottom: 1rem;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

    @st.cache_resource(show_spinner="Loading dataset and trained ML models...")
    def get_cached_pipeline(custom_path: Optional[str] = None, retrain_trigger: int = 0) -> Dict[str, Any]:
        force = retrain_trigger > 0
        return predict_next_trading_session(
            custom_data_path=custom_path,
            force_retrain=force,
        )

    with st.sidebar:
        st.markdown(
            """
            <div style="text-align:center; padding: 0.5rem 0 1rem 0; border-bottom: 1px solid rgba(212,175,55,0.25);">
                <div style="font-size: 2.2rem;">🪙</div>
                <div style="color: #F5D061; font-weight: 700; font-size: 1.15rem;">Gold Price Predictor</div>
                <div style="color: #94A3B8; font-size: 0.8rem;">Supervised ML Regression Suite</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Navigation")
        nav_section = st.radio(
            "Select Application View:",
            options=[
                "A. Dashboard",
                "B. Historical Analysis",
                "C. Prediction",
                "Custom Date Prediction",
                "D. Model Comparison",
                "E. Feature Importance",
                "F. About Project",
            ],
            index=0,
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("### Model & Dataset Management")

        if "retrain_counter" not in st.session_state:
            st.session_state["retrain_counter"] = 0

        if st.button("🔄 Retrain Models on Dataset", width="stretch"):
            st.session_state["retrain_counter"] += 1
            st.cache_resource.clear()
            st.toast("Models successfully retrained and saved with Joblib!", icon="✅")

        st.markdown("---")
        st.markdown("### Currency Unit Reference")
        st.caption(
            "The Kaggle dataset records gold benchmark prices in **USD per Share (SPDR Gold Shares ETF / GLD, ~1/10th troy oz)** "
            "and does not contain historical USD/INR exchange rates. In accordance with strict data integrity, "
            "native USD values are always shown."
        )
        enable_user_inr = st.checkbox(
            "Show Optional User-Supplied INR Reference",
            value=False,
            help="Optionally enter your own known USD-to-INR rate to view converted reference numbers alongside native dataset units.",
        )
        user_usd_inr_rate = 0.0
        if enable_user_inr:
            user_usd_inr_rate = st.number_input(
                "Your Custom USD → INR Exchange Rate (₹ per $1):",
                min_value=1.0,
                max_value=200.0,
                value=86.50,
                step=0.25,
                help="User-supplied exchange rate (never silently assumed by the model).",
            )
            st.caption(
                "Note: 1 GLD ETF share ≈ 0.1 troy oz (3.11035 g). Reference INR figures are strictly a display helper."
            )

    try:
        pipeline_data = get_cached_pipeline(retrain_trigger=st.session_state["retrain_counter"])
    except (DataLoaderError, PreprocessingError, PredictionError, Exception) as err:
        st.error(f"### ⚠️ Application Error\n{err}")
        st.info(
            "**Troubleshooting Tips:**\n"
            "- Ensure your Kaggle CSV file is located inside `data/gold_data.csv` or `Data set/gold_prices_10y.csv`.\n"
            "- Verify the CSV contains valid date and positive numeric gold price columns with at least 50 trading rows."
        )
        st.stop()

    metadata = pipeline_data["metadata"]
    inspection = metadata["inspection_report"]
    split_info = metadata["split_info"]
    unit_info = pipeline_data["unit_info"]
    currency_sym = unit_info["currency_symbol"]
    short_unit = unit_info["short_unit"]
    unit_label = unit_info["unit_label"]

    latest_date = pipeline_data["latest_date"]
    latest_day_name = pipeline_data["latest_day_name"]
    latest_price = pipeline_data["latest_price"]
    best_model_name = pipeline_data["best_model_name"]
    predicted_price = pipeline_data["predicted_price"]
    abs_change = pipeline_data["absolute_change"]
    pct_change = pipeline_data["percentage_change"]
    clean_df = pipeline_data["clean_df"]
    metrics_df = metadata["metrics_df"]
    test_results_df = metadata["test_results_df"]
    feature_importances = metadata["feature_importances"]

    def render_inr_helper_badge(usd_amount: float) -> str:
        if not enable_user_inr or user_usd_inr_rate <= 0:
            return ""
        inr_per_share = usd_amount * user_usd_inr_rate
        inr_per_gram = inr_per_share / 3.11034768
        return (
            f"<div style='margin-top:0.35rem; padding-top:0.35rem; border-top:1px dashed rgba(212,175,55,0.25); "
            f"font-size:0.78rem; color:#FDE68A;'>"
            f"User FX Ref (@ ₹{user_usd_inr_rate:.2f}/$): <b>₹{inr_per_share:,.2f}/share</b> "
            f"(~₹{inr_per_gram:,.0f}/g bullion equiv.)"
            f"</div>"
        )

    # SECTION A: DASHBOARD
    if nav_section == "A. Dashboard":
        st.markdown(
            f"""
            <div class="gold-banner">
                <h1>🪙 Gold Price Prediction Dashboard</h1>
                <p>
                    Supervised Machine Learning Regression system predicting the <b>next available trading session's</b>
                    gold price and custom future horizons from historical market data (<code>{inspection['file_name']}</code>,
                    {inspection['start_date']} to {inspection['end_date']}).
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="unit-notice">
                <b>📌 Verified Dataset Price Unit: {unit_label}</b><br/>
                {unit_info['explanation']}
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Latest Available Price</div>
                    <div class="metric-value">{currency_sym}{latest_price:,.2f}</div>
                    <div class="metric-sub">Session: <b>{latest_date}</b> ({latest_day_name})<br/>Unit: {short_unit}</div>
                    {render_inr_helper_badge(latest_price)}
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Predicted Next Session Price</div>
                    <div class="metric-value">{currency_sym}{predicted_price:,.2f}</div>
                    <div class="metric-sub">Next Trading Day Forecast<br/>Unit: {short_unit}</div>
                    {render_inr_helper_badge(predicted_price)}
                </div>
                """,
                unsafe_allow_html=True,
            )

        delta_class = "delta-pos" if abs_change >= 0 else "delta-neg"
        delta_sign = "+" if abs_change >= 0 else ""
        arrow = "▲" if abs_change >= 0 else "▼"

        with col3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Expected Change vs Latest</div>
                    <div class="metric-value {delta_class}">{delta_sign}{currency_sym}{abs_change:,.2f}</div>
                    <div class="metric-sub">
                        Percentage Change: <span class="{delta_class}">{arrow} {delta_sign}{pct_change:.2f}%</span>
                    </div>
                    {render_inr_helper_badge(abs(abs_change))}
                </div>
                """,
                unsafe_allow_html=True,
            )

        best_r2 = metadata["best_model_metrics"]["R2"]
        best_rmse = metadata["best_model_metrics"]["RMSE"]
        with col4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Selected Best Model</div>
                    <div class="metric-value" style="font-size:1.35rem;">{best_model_name}</div>
                    <div class="metric-sub">
                        Validation R²: <b>{best_r2:.4f}</b> | RMSE: <b>{currency_sym}{best_rmse:.2f}</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br/>", unsafe_allow_html=True)
        left_col, right_col = st.columns([2.1, 1.0])

        with left_col:
            st.subheader("📈 Recent 120 Trading Sessions & Next-Session Forecast")
            recent_df = clean_df.tail(120).copy()
            next_plot_date = pd.to_datetime(latest_date) + pd.tseries.offsets.BDay(1)

            fig_dash = go.Figure()
            fig_dash.add_trace(
                go.Scatter(
                    x=recent_df["Date"],
                    y=recent_df["Close"],
                    mode="lines",
                    name=f"Historical Close ({short_unit})",
                    line=dict(color="#D4AF37", width=2.5),
                )
            )
            fig_dash.add_trace(
                go.Scatter(
                    x=[recent_df["Date"].iloc[-1], next_plot_date],
                    y=[latest_price, predicted_price],
                    mode="lines+markers",
                    name=f"Forecast ({best_model_name})",
                    line=dict(color="#34D399" if abs_change >= 0 else "#F87171", width=2.5, dash="dot"),
                    marker=dict(size=[7, 12], symbol=["circle", "star"]),
                )
            )
            fig_dash.update_layout(
                template="plotly_dark",
                paper_bgcolor="#141B26",
                plot_bgcolor="#141B26",
                height=410,
                margin=dict(l=20, r=20, t=25, b=20),
                xaxis_title="Trading Session Date",
                yaxis_title=f"Gold Price ({unit_label})",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified",
            )
            st.plotly_chart(fig_dash, width="stretch")

        with right_col:
            st.subheader("🔍 Dataset Inspection Summary")
            st.markdown(
                f"""
                <div class="section-card">
                    <p style="margin:0 0 0.55rem 0;"><b>File:</b> <code>{inspection['file_name']}</code></p>
                    <p style="margin:0 0 0.55rem 0;"><b>Total Trading Sessions:</b> {inspection['num_rows']:,} rows</p>
                    <p style="margin:0 0 0.55rem 0;"><b>Columns Detected:</b> {inspection['num_columns']} columns</p>
                    <p style="margin:0 0 0.55rem 0;"><b>Date Column:</b> <code>{inspection['date_column']}</code></p>
                    <p style="margin:0 0 0.55rem 0;"><b>Target Price Column:</b> <code>{inspection['price_column']}</code></p>
                    <p style="margin:0 0 0.55rem 0;"><b>Date Range:</b> {inspection['start_date']} → {inspection['end_date']}</p>
                    <p style="margin:0 0 0.55rem 0;"><b>Observation Frequency:</b> {inspection['frequency_info']['frequency']}</p>
                    <p style="margin:0 0 0.55rem 0;"><b>Duplicate Dates / Invalid Prices:</b> {inspection['duplicate_dates']} / {inspection['invalid_or_nonpositive_prices']}</p>
                    <p style="margin:0;"><b>Historical Range:</b> {currency_sym}{inspection['price_min']:,.2f} – {currency_sym}{inspection['price_max']:,.2f}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # SECTION B: HISTORICAL ANALYSIS
    elif nav_section == "B. Historical Analysis":
        st.markdown(
            """
            <div class="gold-banner">
                <h1>📊 Historical Gold Price Analysis</h1>
                <p>Explore 10 years of historical gold trading sessions across Daily, Weekly, and Monthly trend views.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        resampled_views = get_resampled_trends(clean_df)

        ctrl1, ctrl2, ctrl3 = st.columns([1.2, 1.4, 1.4])
        with ctrl1:
            trend_granularity = st.selectbox(
                "Select Time Horizon / Frequency View:",
                options=["Daily (Trading Sessions)", "Weekly (Week-Ending Friday)", "Monthly (Month-End)"],
                index=0,
            )
        with ctrl2:
            min_d = clean_df["Date"].min().date()
            max_d = clean_df["Date"].max().date()
            date_range = st.date_input(
                "Filter Date Range:",
                value=(min_d, max_d),
                min_value=min_d,
                max_value=max_d,
            )
        with ctrl3:
            show_mas = st.multiselect(
                "Overlay Moving Averages (Daily View):",
                options=["7-Session MA", "14-Session MA", "50-Session MA", "200-Session MA"],
                default=["50-Session MA", "200-Session MA"],
            )

        if trend_granularity.startswith("Daily"):
            view_df = resampled_views["Daily"].copy()
            view_df["7-Session MA"] = view_df["Close"].rolling(7).mean()
            view_df["14-Session MA"] = view_df["Close"].rolling(14).mean()
            view_df["50-Session MA"] = view_df["Close"].rolling(50).mean()
            view_df["200-Session MA"] = view_df["Close"].rolling(200).mean()
            freq_title = "Daily Trading Sessions"
        elif trend_granularity.startswith("Weekly"):
            view_df = resampled_views["Weekly"].copy()
            freq_title = "Weekly Resampled (Friday Close)"
        else:
            view_df = resampled_views["Monthly"].copy()
            freq_title = "Monthly Resampled (Month-End Close)"

        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            start_filter, end_filter = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
            view_df = view_df[(view_df["Date"] >= start_filter) & (view_df["Date"] <= end_filter)].copy()

        s1, s2, s3, s4 = st.columns(4)
        if not view_df.empty:
            period_start_p = float(view_df["Close"].iloc[0])
            period_end_p = float(view_df["Close"].iloc[-1])
            period_return = ((period_end_p - period_start_p) / period_start_p) * 100.0
            s1.metric("Observations in View", f"{len(view_df):,}")
            s2.metric(f"Period Low ({short_unit})", f"{currency_sym}{view_df['Close'].min():,.2f}")
            s3.metric(f"Period High ({short_unit})", f"{currency_sym}{view_df['Close'].max():,.2f}")
            s4.metric("Cumulative Period Change", f"{currency_sym}{period_end_p - period_start_p:+,.2f}", f"{period_return:+.2f}%")

        fig_hist = go.Figure()
        fig_hist.add_trace(
            go.Scatter(
                x=view_df["Date"],
                y=view_df["Close"],
                mode="lines",
                name=f"Gold Close ({short_unit})",
                line=dict(color="#D4AF37", width=2.2),
            )
        )

        ma_colors = {
            "7-Session MA": "#38BDF8",
            "14-Session MA": "#A78BFA",
            "50-Session MA": "#34D399",
            "200-Session MA": "#F87171",
        }
        if trend_granularity.startswith("Daily"):
            for ma_name in show_mas:
                if ma_name in view_df.columns:
                    fig_hist.add_trace(
                        go.Scatter(
                            x=view_df["Date"],
                            y=view_df[ma_name],
                            mode="lines",
                            name=ma_name,
                            line=dict(color=ma_colors.get(ma_name, "#94A3B8"), width=1.6, dash="dash"),
                        )
                    )

        fig_hist.update_layout(
            title=f"Historical Gold Price Trend — {freq_title} ({unit_label})",
            template="plotly_dark",
            paper_bgcolor="#141B26",
            plot_bgcolor="#141B26",
            height=460,
            xaxis_title="Date",
            yaxis_title=f"Gold Price ({unit_label})",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
        )
        st.plotly_chart(fig_hist, width="stretch")

        st.subheader("🗂️ Dataset Preview & Schema Inspection")
        tab_preview, tab_schema = st.tabs(["📄 Historical Dataset Preview", "🔍 Column & Missing Value Audit"])

        with tab_preview:
            st.caption(
                f"Showing cleaned dataset records from `{inspection['file_name']}` "
                f"({len(view_df):,} observations in selected view). Sorted chronologically."
            )
            st.dataframe(
                view_df.sort_values("Date", ascending=False).head(100),
                width="stretch",
                hide_index=True,
            )

        with tab_schema:
            schema_rows = []
            for col in inspection["columns"]:
                schema_rows.append({
                    "Column Name": col,
                    "Data Type": inspection["dtypes"].get(col, ""),
                    "Missing Values (Raw CSV)": inspection["missing_by_column"].get(col, 0),
                    "Role in Pipeline": (
                        "Primary Date Index" if col == inspection["date_column"]
                        else "Primary Gold Price (Target / Lag Source)" if col == inspection["price_column"]
                        else "Pre-computed CSV Indicator (Re-engineered causally in pipeline)"
                    ),
                })
            st.dataframe(pd.DataFrame(schema_rows), width="stretch", hide_index=True)

    # SECTION C: PREDICTION (NEXT TRADING SESSION)
    elif nav_section == "C. Prediction":
        st.markdown(
            """
            <div class="gold-banner">
                <h1>🎯 Next Trading Session Gold Price Prediction</h1>
                <p>Forecasting the next available market trading session using the latest historical features.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.warning(
            f"**📅 Important Trading Calendar Notice:**\n\n"
            f"- **Latest Available Observation Date:** `{latest_date}` (**{latest_day_name}**)\n"
            f"- **Target Horizon:** This prediction is for the **next available trading session** when the market opens, "
            f"**not necessarily the next calendar day** (financial markets do not trade on Saturdays, Sundays, or official market holidays)."
        )

        model_options = list(pipeline_data["all_model_predictions"].keys())
        default_idx = model_options.index(best_model_name) if best_model_name in model_options else 0

        selected_pred_model = st.selectbox(
            "Select Regression Model to Inspect Forecast (Default = Best Chronological Validation Model):",
            options=model_options,
            index=default_idx,
        )

        chosen_pred_info = pipeline_data["all_model_predictions"][selected_pred_model]
        c_pred = chosen_pred_info["predicted_price"]
        c_abs = chosen_pred_info["absolute_change"]
        c_pct = chosen_pred_info["percentage_change"]
        c_class = "delta-pos" if c_abs >= 0 else "delta-neg"
        c_sign = "+" if c_abs >= 0 else ""

        pcol1, pcol2, pcol3 = st.columns(3)
        with pcol1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Latest Observed Price ({latest_date})</div>
                    <div class="metric-value">{currency_sym}{latest_price:,.2f}</div>
                    <div class="metric-sub">Unit: {unit_label}</div>
                    {render_inr_helper_badge(latest_price)}
                </div>
                """,
                unsafe_allow_html=True,
            )
        with pcol2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Predicted Next Trading Session Price</div>
                    <div class="metric-value">{currency_sym}{c_pred:,.2f}</div>
                    <div class="metric-sub">Model: <b>{selected_pred_model}</b></div>
                    {render_inr_helper_badge(c_pred)}
                </div>
                """,
                unsafe_allow_html=True,
            )
        with pcol3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Projected Session Move</div>
                    <div class="metric-value {c_class}">{c_sign}{currency_sym}{c_abs:,.2f} ({c_sign}{c_pct:.2f}%)</div>
                    <div class="metric-sub">Best Validated Model: <b>{best_model_name}</b></div>
                    {render_inr_helper_badge(abs(c_abs))}
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br/>", unsafe_allow_html=True)

        f_col1, f_col2 = st.columns([1.15, 1.0])
        with f_col1:
            st.subheader("🧮 Latest Historical Feature Inputs Used at Prediction Time")
            feat_rows = []
            for f_name, f_val in pipeline_data["latest_features"].items():
                formatted_val = f"{f_val:+.3f}%" if "Pct" in f_name else (
                    f"{f_val:.3f}%" if "Volatility" in f_name else f"{currency_sym}{f_val:,.2f}"
                )
                feat_rows.append({
                    "Feature Name": f_name,
                    f"Value on {latest_date}": formatted_val,
                    "Meaning": metadata["feature_descriptions"].get(f_name, ""),
                })
            st.dataframe(pd.DataFrame(feat_rows), width="stretch", hide_index=True)

        with f_col2:
            st.subheader("🤖 Next-Session Forecast Across All Models")
            all_preds_rows = []
            for m_name, p_dict in pipeline_data["all_model_predictions"].items():
                all_preds_rows.append({
                    "Model": f"⭐ {m_name}" if m_name == best_model_name else m_name,
                    f"Predicted Price ({short_unit})": f"{currency_sym}{p_dict['predicted_price']:,.2f}",
                    "Abs Diff": f"{p_dict['absolute_change']:+,.2f}",
                    "Pct Diff (%)": f"{p_dict['percentage_change']:+.2f}%",
                })
            st.dataframe(pd.DataFrame(all_preds_rows), width="stretch", hide_index=True)

    # CUSTOM DATE PREDICTION SECTION
    elif nav_section == "Custom Date Prediction":
        st.markdown(
            f"""
            <div class="gold-banner">
                <h1>📅 Custom Future Date Gold Price Prediction</h1>
                <p>
                    Select any future date up to <b>{MAX_FORECAST_CALENDAR_DAYS} calendar days</b> after the latest
                    historical dataset observation (<code>{latest_date}</code>) to generate a multi-step recursive gold price forecast.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        latest_dt_obj = pd.to_datetime(latest_date).date()
        min_future_date = latest_dt_obj + timedelta(days=1)
        max_future_date = latest_dt_obj + timedelta(days=MAX_FORECAST_CALENDAR_DAYS)
        next_trading_dt = (pd.to_datetime(latest_date) + pd.tseries.offsets.BDay(1)).date()

        if "custom_target_date" not in st.session_state:
            st.session_state["custom_target_date"] = latest_dt_obj + timedelta(days=30)

        st.markdown("##### ⚡ Quick Horizon Presets (or Pick Any Date Below)")
        q1, q2, q3, q4, q5 = st.columns(5)
        if q1.button("Next Trading Day", width="stretch"):
            st.session_state["custom_target_date"] = next_trading_dt
        if q2.button("+7 Calendar Days", width="stretch"):
            st.session_state["custom_target_date"] = latest_dt_obj + timedelta(days=7)
        if q3.button("+30 Calendar Days", width="stretch"):
            st.session_state["custom_target_date"] = latest_dt_obj + timedelta(days=30)
        if q4.button("+90 Calendar Days", width="stretch"):
            st.session_state["custom_target_date"] = latest_dt_obj + timedelta(days=90)
        if q5.button("+1 Year (365 Days)", width="stretch"):
            st.session_state["custom_target_date"] = max_future_date

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        form_col1, form_col2, form_col3, form_col4 = st.columns([1.3, 1.3, 1.1, 1.0])

        with form_col1:
            selected_future_date = st.date_input(
                "1️⃣ Select Future Target Date (Day / Month / Year):",
                value=st.session_state["custom_target_date"],
                min_value=min_future_date,
                max_value=max_future_date,
            )
            st.session_state["custom_target_date"] = selected_future_date

        with form_col2:
            ml_model_choices = list(metadata["trained_models"].keys())
            default_ml_idx = ml_model_choices.index(best_model_name) if best_model_name in ml_model_choices else 0
            selected_custom_model = st.selectbox(
                "2️⃣ Select Forecasting Model:",
                options=ml_model_choices,
                index=default_ml_idx,
            )

        with form_col3:
            history_lookback_label = st.selectbox(
                "3️⃣ Historical Context on Chart:",
                options=[
                    "Last 90 Trading Sessions",
                    "Last 180 Trading Sessions",
                    "Last 1 Year (252 Sessions)",
                    "Full 10-Year History",
                ],
                index=1,
            )

        with form_col4:
            st.markdown("<div style='height: 1.65rem;'></div>", unsafe_allow_html=True)
            predict_clicked = st.button(
                "🔮 Predict Gold Price",
                type="primary",
                width="stretch",
            )
        st.markdown("</div>", unsafe_allow_html=True)

        sel_cal_days = (selected_future_date - latest_dt_obj).days
        sel_day_of_week = selected_future_date.strftime("%A")
        st.markdown(
            f"**📌 Selected Target Date:** `{selected_future_date.strftime('%Y-%m-%d')}` (**{sel_day_of_week}**) — "
            f"**{sel_cal_days} calendar day{'s' if sel_cal_days != 1 else ''}** after latest historical observation (`{latest_date}`)."
        )

        if predict_clicked:
            st.toast(f"Generated multi-step forecast for {selected_future_date.strftime('%Y-%m-%d')}!", icon="🔮")

        try:
            custom_forecast = predict_custom_future_date(
                target_date=selected_future_date,
                selected_model_name=selected_custom_model,
            )
        except PredictionError as pred_err:
            st.error(f"### ⚠️ Date Prediction Error\n{pred_err}")
            st.stop()

        f_pred_price = custom_forecast["predicted_price"]
        f_abs_change = custom_forecast["absolute_change"]
        f_pct_change = custom_forecast["percentage_change"]
        f_cal_days = custom_forecast["calendar_days"]
        f_trad_days = custom_forecast["estimated_trading_sessions"]
        f_eff_date = custom_forecast["effective_trading_date"]
        f_eff_day = custom_forecast["effective_day_name"]
        f_is_weekend = custom_forecast["is_weekend_target"]
        f_df = custom_forecast["forecast_df"]

        f_delta_class = "delta-pos" if f_abs_change >= 0 else "delta-neg"
        f_delta_sign = "+" if f_abs_change >= 0 else ""
        f_arrow = "▲" if f_abs_change >= 0 else "▼"

        badge_type = custom_forecast["badge_color"]
        warn_msg = custom_forecast["warning_message"]
        if badge_type == "info":
            st.info(f"**🛡️ Forecast Horizon & Uncertainty Assessment — {custom_forecast['uncertainty_level']}**\n\n{warn_msg}")
        elif badge_type == "warning":
            st.warning(f"**⚠️ Forecast Horizon & Uncertainty Assessment — {custom_forecast['uncertainty_level']}**\n\n{warn_msg}")
        else:
            st.error(f"**🚨 Forecast Horizon & Uncertainty Assessment — {custom_forecast['uncertainty_level']}**\n\n{warn_msg}")

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            weekend_sub = (
                f"Weekend → Trading Day: <b>{f_eff_date}</b> ({f_eff_day})"
                if f_is_weekend
                else f"Trading Session ({sel_day_of_week})"
            )
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Selected Target Date</div>
                    <div class="metric-value" style="font-size:1.45rem;">{custom_forecast['selected_date']}</div>
                    <div class="metric-sub">
                        {weekend_sub}<br/>
                        Horizon: <b>{f_cal_days} Cal. Days</b> (~<b>{f_trad_days} Trading Sessions</b>)
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with k2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Latest Historical Price ({latest_date})</div>
                    <div class="metric-value">{currency_sym}{latest_price:,.2f}</div>
                    <div class="metric-sub">Last Observed Close ({latest_day_name})<br/>Unit: {short_unit}</div>
                    {render_inr_helper_badge(latest_price)}
                </div>
                """,
                unsafe_allow_html=True,
            )

        with k3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Predicted Price on {custom_forecast['selected_date']}</div>
                    <div class="metric-value">{currency_sym}{f_pred_price:,.2f}</div>
                    <div class="metric-sub">
                        68% Range: {currency_sym}{custom_forecast['lower_68']:,.2f} – {currency_sym}{custom_forecast['upper_68']:,.2f}
                    </div>
                    {render_inr_helper_badge(f_pred_price)}
                </div>
                """,
                unsafe_allow_html=True,
            )

        with k4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Projected Difference vs Latest</div>
                    <div class="metric-value {f_delta_class}">{f_delta_sign}{currency_sym}{f_abs_change:,.2f}</div>
                    <div class="metric-sub">
                        Pct Change: <span class="{f_delta_class}">{f_arrow} {f_delta_sign}{f_pct_change:.2f}%</span><br/>
                        Model: <b>{custom_forecast['active_model_name']}</b>
                    </div>
                    {render_inr_helper_badge(abs(f_abs_change))}
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br/>", unsafe_allow_html=True)
        st.subheader("📈 Historical Gold Prices & Multi-Step Forecast Trajectory")

        if history_lookback_label.startswith("Last 90"):
            hist_slice = clean_df.tail(90).copy()
        elif history_lookback_label.startswith("Last 180"):
            hist_slice = clean_df.tail(180).copy()
        elif history_lookback_label.startswith("Last 1 Year"):
            hist_slice = clean_df.tail(252).copy()
        else:
            hist_slice = clean_df.copy()

        connect_dates = [pd.to_datetime(latest_date)] + list(f_df["Trading_Date"])
        connect_preds = [latest_price] + list(f_df["Predicted_Price"])
        connect_upper95 = [latest_price] + list(f_df["Upper_95"])
        connect_lower95 = [latest_price] + list(f_df["Lower_95"])
        connect_upper68 = [latest_price] + list(f_df["Upper_68"])
        connect_lower68 = [latest_price] + list(f_df["Lower_68"])

        fig_future = go.Figure()
        fig_future.add_trace(
            go.Scatter(
                x=connect_dates + connect_dates[::-1],
                y=connect_upper95 + connect_lower95[::-1],
                fill="toself",
                fillcolor="rgba(56, 189, 248, 0.10)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=True,
                name="95% Uncertainty Cone (±1.96σ√h)",
            )
        )
        fig_future.add_trace(
            go.Scatter(
                x=connect_dates + connect_dates[::-1],
                y=connect_upper68 + connect_lower68[::-1],
                fill="toself",
                fillcolor="rgba(56, 189, 248, 0.20)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=True,
                name="68% Uncertainty Cone (±1σ√h)",
            )
        )
        fig_future.add_trace(
            go.Scatter(
                x=hist_slice["Date"],
                y=hist_slice["Close"],
                mode="lines",
                name=f"Historical Gold Price ({short_unit})",
                line=dict(color="#D4AF37", width=2.5),
            )
        )
        fig_future.add_trace(
            go.Scatter(
                x=connect_dates,
                y=connect_preds,
                mode="lines",
                name=f"Forecasted Trajectory ({custom_forecast['active_model_name']})",
                line=dict(color="#38BDF8", width=2.6, dash="dash"),
            )
        )
        target_plot_dt = pd.to_datetime(f_eff_date)
        fig_future.add_trace(
            go.Scatter(
                x=[target_plot_dt],
                y=[f_pred_price],
                mode="markers+text",
                name=f"Target Date Forecast ({custom_forecast['selected_date']}: {currency_sym}{f_pred_price:,.2f})",
                text=[f"  {custom_forecast['selected_date']}: {currency_sym}{f_pred_price:,.2f}"],
                textposition="top left",
                marker=dict(
                    color="#34D399" if f_abs_change >= 0 else "#F87171",
                    size=14,
                    symbol="star",
                    line=dict(color="#FFFFFF", width=1.5),
                ),
            )
        )
        fig_future.update_layout(
            title=f"Historical vs. Forecasted Gold Price — Target: {custom_forecast['selected_date']}",
            template="plotly_dark",
            paper_bgcolor="#141B26",
            plot_bgcolor="#141B26",
            height=500,
            xaxis_title="Date",
            yaxis_title=f"Gold Price ({unit_label})",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
        )
        st.plotly_chart(fig_future, width="stretch")

        tab_traj, tab_method = st.tabs([
            f"📋 Step-by-Step Forecast Table ({f_trad_days} Trading Sessions)",
            "🧠 How Recursive Multi-Step Forecasting Works & Its Limitations",
        ])

        with tab_traj:
            traj_display = f_df[[
                "Step",
                "Trading_Date_Str",
                "Predicted_Price",
                "Abs_Change_From_Latest",
                "Pct_Change_From_Latest",
                "Lower_68",
                "Upper_68",
                "Lower_95",
                "Upper_95",
            ]].rename(columns={
                "Step": "Session Step (h)",
                "Trading_Date_Str": "Estimated Trading Date",
                "Predicted_Price": f"Predicted Price ({short_unit})",
                "Abs_Change_From_Latest": f"Diff vs {latest_date} ({currency_sym})",
                "Pct_Change_From_Latest": "Pct Change (%)",
                "Lower_68": "68% Lower",
                "Upper_68": "68% Upper",
                "Lower_95": "95% Lower",
                "Upper_95": "95% Upper",
            })
            st.dataframe(traj_display, width="stretch", hide_index=True)

        with tab_method:
            st.markdown(
                r"""
                #### 1. Why We Cannot Use the Same 1-Day Prediction for Every Future Date
                A 1-step supervised regression model is trained to predict **tomorrow's** trading session ($P_{t+1}$) from **today's** known lags ($P_t, P_{t-1}, \dots, P_{t-7}$). If you asked for a date 30 or 180 days in the future and simply returned $\hat{P}_{t+1}$, every future date would show the exact same number, ignoring the multi-day trajectory of the model.

                #### 2. How Recursive Multi-Step Forecasting Works
                - **Step $h = 1$ (Next Trading Session):** Feeds the latest actual historical prices (`2026-01-23` and prior sessions) into the trained model to predict $\hat{P}_{t+1}$.
                - **Steps $h = 2 \dots H$:** Appends each predicted price to the chronological price series and recomputes `Lag_1..7`, `MA_7`, `MA_14`, `Pct_Change`, and `Volatility_7` step by step.
                """
            )

    # SECTION D: MODEL COMPARISON
    elif nav_section == "D. Model Comparison":
        st.markdown(
            """
            <div class="gold-banner">
                <h1>🏆 Supervised Regression Model Comparison</h1>
                <p>Chronological validation performance of all 4 Supervised Machine Learning algorithms compared against the Persistence Baseline.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        display_metrics = metrics_df.copy()
        display_metrics["Status"] = display_metrics["Model"].apply(
            lambda m: "🏆 Best ML Model" if m == best_model_name else ("📏 Benchmark" if "Baseline" in m else "Evaluated")
        )
        display_cols = ["Status", "Model", "Type", "MAE", "MSE", "RMSE", "R2", "Train_RMSE", "Train_R2"]
        st.dataframe(display_metrics[display_cols], width="stretch", hide_index=True)

        fig_comp = go.Figure()
        fig_comp.add_trace(
            go.Scatter(
                x=test_results_df["Target_Date"],
                y=test_results_df["Actual_Next_Close"],
                mode="lines",
                name="Actual Gold Price",
                line=dict(color="#F5D061", width=3),
            )
        )
        fig_comp.add_trace(
            go.Scatter(
                x=test_results_df["Target_Date"],
                y=test_results_df["Linear Regression"],
                mode="lines",
                name="Predicted — Linear Regression",
                line=dict(color="#38BDF8", width=2, dash="dot"),
            )
        )
        fig_comp.update_layout(template="plotly_dark", paper_bgcolor="#141B26", plot_bgcolor="#141B26", height=460)
        st.plotly_chart(fig_comp, width="stretch")

    # SECTION E: FEATURE IMPORTANCE
    elif nav_section == "E. Feature Importance":
        st.markdown(
            """
            <div class="gold-banner">
                <h1>🌲 Tree-Based Feature Importance Analysis</h1>
                <p>Inspect which historical lag and moving-average features contribute most to tree-based splits.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        tree_model_names = list(feature_importances.keys())
        selected_tree_model = st.selectbox("Select Tree-Based Regression Algorithm:", options=tree_model_names)
        imp_df = feature_importances[selected_tree_model]
        fig_imp = px.bar(
            imp_df.sort_values("Importance_Pct", ascending=True),
            x="Importance_Pct",
            y="Feature",
            orientation="h",
            text_auto=".2f",
            title=f"Relative Feature Importance (%) — {selected_tree_model}",
        )
        fig_imp.update_layout(template="plotly_dark", paper_bgcolor="#141B26", plot_bgcolor="#141B26", height=420)
        st.plotly_chart(fig_imp, width="stretch")

    # SECTION F: ABOUT PROJECT
    elif nav_section == "F. About Project":
        st.markdown(
            """
            <div class="gold-banner">
                <h1>📘 About Project — Educational Guide</h1>
                <p>Complete conceptual walkthrough of Supervised Machine Learning, Time-Series Regression, Multi-Step Forecasting, and System Architecture.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            r"""
            ### 1. Supervised Learning & Regression
            Learns a continuous mapping $f(X_t) \rightarrow P_{t+1}$ from historical lag features (`Current_Close`, `Lag_1..7`, `MA_7`, `MA_14`, `Pct_Change`, `Pct_Change_7`, `Volatility_7`) to the next available trading session's closing price.

            ### 2. Chronological Splitting & Data Leakage Prevention
            Trains on the oldest 80% of trading sessions (`2016–2024`) and validates on the newest 20% (`2024–2026`) without shuffling (`shuffle=False`), fitting `StandardScaler` strictly on training data.
            """
        )


if _is_running_in_streamlit():
    render_streamlit_dashboard()
