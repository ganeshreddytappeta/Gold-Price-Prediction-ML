"""
app.py
------
Streamlit Web Application for Gold Price Prediction Using Supervised Machine Learning.
Features a professional Gold & Dark financial dashboard theme with sections:
- A. Dashboard
- B. Historical Analysis
- C. Prediction (Next Trading Session)
- Custom Date Prediction (Multi-Step Recursive Forecasting up to 365 Days Ahead)
- D. Model Comparison
- E. Feature Importance
- F. About Project
"""

from datetime import timedelta
from pathlib import Path
from typing import Dict, Any, Optional
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

import importlib
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


# ---------------------------------------------------------------------------
# Page Configuration & Custom Gold/Dark Theme Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Gold Price Prediction | Supervised ML Dashboard",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    /* Main background and typography */
    .stApp {
        background-color: #0B0F17;
        color: #F3F4F6;
    }

    /* Header Banner */
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

    /* Metric Cards */
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

    /* Info / Notice boxes */
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
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Caching & Data Loading Helpers
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading dataset and trained ML models...")
def get_cached_pipeline(custom_path: Optional[str] = None, retrain_trigger: int = 0) -> Dict[str, Any]:
    """Load or train the ML pipeline and return all prediction and evaluation artifacts."""
    force = retrain_trigger > 0
    return predict_next_trading_session(
        custom_data_path=custom_path,
        force_retrain=force,
    )


# ---------------------------------------------------------------------------
# Sidebar Navigation & Controls
# ---------------------------------------------------------------------------
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

    # Optional User-Provided INR Reference Calculator (never silently invents rates)
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


# ---------------------------------------------------------------------------
# Load Artifacts with Graceful Error Handling
# ---------------------------------------------------------------------------
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
preproc_info = metadata["preprocessing_summary"]
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
    """Return an optional user-supplied INR conversion caption if enabled by the user."""
    if not enable_user_inr or user_usd_inr_rate <= 0:
        return ""
    inr_per_share = usd_amount * user_usd_inr_rate
    # 1 GLD share ~ 1/10th troy oz = 3.11034768 grams
    inr_per_gram = inr_per_share / 3.11034768
    return (
        f"<div style='margin-top:0.35rem; padding-top:0.35rem; border-top:1px dashed rgba(212,175,55,0.25); "
        f"font-size:0.78rem; color:#FDE68A;'>"
        f"User FX Ref (@ ₹{user_usd_inr_rate:.2f}/$): <b>₹{inr_per_share:,.2f}/share</b> "
        f"(~₹{inr_per_gram:,.0f}/g bullion equiv.)"
        f"</div>"
    )


# ===========================================================================
# SECTION A: DASHBOARD
# ===========================================================================
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

    # Dataset Price Unit Transparency Banner
    st.markdown(
        f"""
        <div class="unit-notice">
            <b>📌 Verified Dataset Price Unit: {unit_label}</b><br/>
            {unit_info['explanation']}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Top 4 KPI Metric Cards
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

    # Recent 120 Trading Sessions + Next Session Forecast Point
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
        # Connect latest point to predicted next session
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


# ===========================================================================
# SECTION B: HISTORICAL ANALYSIS
# ===========================================================================
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

    # Summary stats row for selected date range
    s1, s2, s3, s4 = st.columns(4)
    if not view_df.empty:
        period_start_p = float(view_df["Close"].iloc[0])
        period_end_p = float(view_df["Close"].iloc[-1])
        period_return = ((period_end_p - period_start_p) / period_start_p) * 100.0
        s1.metric("Observations in View", f"{len(view_df):,}")
        s2.metric(f"Period Low ({short_unit})", f"{currency_sym}{view_df['Close'].min():,.2f}")
        s3.metric(f"Period High ({short_unit})", f"{currency_sym}{view_df['Close'].max():,.2f}")
        s4.metric("Cumulative Period Change", f"{currency_sym}{period_end_p - period_start_p:+,.2f}", f"{period_return:+.2f}%")

    # Interactive Historical Price Line Chart
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

    # Dataset Preview & Column Audit
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
        st.caption(
            "**Note on Raw CSV Missing Values:** In `gold_prices_10y.csv`, core OHLCV columns (`Date`, `Close`, `High`, "
            "`Low`, `Open`, `Volume`, `Adj Close`) have **0 missing values**. Only the pre-computed rolling columns "
            "(`Daily_Return`, `MA_20`, `MA_50`, `MA_200`, `Volatility_20`) have initial warm-up NaNs on the first "
            "1 to 199 rows. Our pipeline computes its own strictly causal 7- and 14-session features directly from `Close`."
        )


# ===========================================================================
# SECTION C: PREDICTION (NEXT TRADING SESSION)
# ===========================================================================
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

    # Explicit Trading Observation vs Calendar Day Notice
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
        st.caption(
            f"All features below are computed strictly from observations on or before `{latest_date}` (Zero Data Leakage)."
        )
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

        st.info(
            "**💡 Why do Tree-Based Models predict ~$188 while Linear Regression predicts ~$456.79?**\n\n"
            "During the chronological training period (`2016–2024`), gold prices ranged between **$106.95 and ~$199.00**. "
            "Between `2024` and `2026`, gold rallied to new all-time highs (**$458.00**).\n\n"
            "- **Decision Trees, Random Forests, and Gradient Boosting** predict averages of training leaf nodes, "
            "so they mathematically **cannot extrapolate above the maximum price seen in training (~$199)**.\n"
            "- **Linear Regression** learns linear relationships (`Next_Price ≈ w_1 × Current_Close + ...`), "
            "allowing it to extrapolate accurately into new all-time-high price regimes!"
        )

    st.subheader("⚠️ Prediction Limitations & Financial Disclaimer")
    st.markdown(
        """
        <div class="section-card">
            <ul style="margin:0; padding-left:1.2rem; line-height:1.65;">
                <li><b>Trading Day vs. Calendar Day:</b> Predictions apply strictly to the next open market session, not weekends or holidays.</li>
                <li><b>Unobserved External Shocks:</b> Supervised models trained solely on historical price lags cannot anticipate central bank rate decisions, inflation surprises, geopolitical conflicts, or sudden currency movements.</li>
                <li><b>Non-Stationarity of Financial Series:</b> Gold prices undergo structural regime shifts over decades; past statistical relationships do not guarantee future performance.</li>
                <li><b>Educational Purpose Only:</b> This application is an academic B.Tech Machine Learning project and must not be used as financial or investment advice.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ===========================================================================
# NEW SECTION: CUSTOM DATE PREDICTION (MULTI-STEP RECURSIVE FORECASTING)
# ===========================================================================
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

    # Quick Horizon Preset Buttons for easy testing of 1-session, 7d, 30d, 90d, and 365d horizons
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

    # 1. Date Selection Interface & Model Selector
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    form_col1, form_col2, form_col3, form_col4 = st.columns([1.3, 1.3, 1.1, 1.0])

    with form_col1:
        selected_future_date = st.date_input(
            "1️⃣ Select Future Target Date (Day / Month / Year):",
            value=st.session_state["custom_target_date"],
            min_value=min_future_date,
            max_value=max_future_date,
            help=f"Select any future date from {min_future_date} to {max_future_date} (up to 365 days ahead). Past dates are disabled.",
        )
        st.session_state["custom_target_date"] = selected_future_date

    with form_col2:
        ml_model_choices = list(metadata["trained_models"].keys())
        default_ml_idx = ml_model_choices.index(best_model_name) if best_model_name in ml_model_choices else 0
        selected_custom_model = st.selectbox(
            "2️⃣ Select Forecasting Model:",
            options=ml_model_choices,
            index=default_ml_idx,
            help="Linear Regression is recommended for multi-step forecasting because tree-based models cannot extrapolate beyond training price bounds.",
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

    # Clear summary banner of the user's selected date prior to / alongside results
    sel_cal_days = (selected_future_date - latest_dt_obj).days
    sel_day_of_week = selected_future_date.strftime("%A")
    st.markdown(
        f"**📌 Selected Target Date:** `{selected_future_date.strftime('%Y-%m-%d')}` (**{sel_day_of_week}**) — "
        f"**{sel_cal_days} calendar day{'s' if sel_cal_days != 1 else ''}** after latest historical observation (`{latest_date}`)."
    )

    if predict_clicked:
        st.toast(f"Generated multi-step forecast for {selected_future_date.strftime('%Y-%m-%d')}!", icon="🔮")

    # Run Custom Future Date Forecast with Error Handling
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

    # Uncertainty Warning Box (Dynamically styled by horizon length)
    badge_type = custom_forecast["badge_color"]
    warn_msg = custom_forecast["warning_message"]
    if badge_type == "info":
        st.info(f"**🛡️ Forecast Horizon & Uncertainty Assessment — {custom_forecast['uncertainty_level']}**\n\n{warn_msg}")
    elif badge_type == "warning":
        st.warning(f"**⚠️ Forecast Horizon & Uncertainty Assessment — {custom_forecast['uncertainty_level']}**\n\n{warn_msg}")
    else:
        st.error(f"**🚨 Forecast Horizon & Uncertainty Assessment — {custom_forecast['uncertainty_level']}**\n\n{warn_msg}")

    # 4. Results Display KPI Cards
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

    # 5. Forecast Visualization (Historical + Recursive Forecast + Target Marker + Uncertainty Fan)
    st.subheader("📈 Historical Gold Prices & Multi-Step Forecast Trajectory")

    if history_lookback_label.startswith("Last 90"):
        hist_slice = clean_df.tail(90).copy()
    elif history_lookback_label.startswith("Last 180"):
        hist_slice = clean_df.tail(180).copy()
    elif history_lookback_label.startswith("Last 1 Year"):
        hist_slice = clean_df.tail(252).copy()
    else:
        hist_slice = clean_df.copy()

    # Prepend the latest historical point to the forecast series so the line connects seamlessly
    connect_dates = [pd.to_datetime(latest_date)] + list(f_df["Trading_Date"])
    connect_preds = [latest_price] + list(f_df["Predicted_Price"])
    connect_upper95 = [latest_price] + list(f_df["Upper_95"])
    connect_lower95 = [latest_price] + list(f_df["Lower_95"])
    connect_upper68 = [latest_price] + list(f_df["Upper_68"])
    connect_lower68 = [latest_price] + list(f_df["Lower_68"])

    fig_future = go.Figure()

    # 95% Uncertainty Band
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

    # 68% Uncertainty Band
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

    # Historical Gold Prices
    fig_future.add_trace(
        go.Scatter(
            x=hist_slice["Date"],
            y=hist_slice["Close"],
            mode="lines",
            name=f"Historical Gold Price ({short_unit})",
            line=dict(color="#D4AF37", width=2.5),
        )
    )

    # Forecasted Trajectory Line
    fig_future.add_trace(
        go.Scatter(
            x=connect_dates,
            y=connect_preds,
            mode="lines",
            name=f"Forecasted Trajectory ({custom_forecast['active_model_name']})",
            line=dict(color="#38BDF8", width=2.6, dash="dash"),
        )
    )

    # Latest Historical Observation Marker
    fig_future.add_trace(
        go.Scatter(
            x=[pd.to_datetime(latest_date)],
            y=[latest_price],
            mode="markers",
            name=f"Latest Historical Close ({latest_date}: {currency_sym}{latest_price:,.2f})",
            marker=dict(color="#F5D061", size=10, symbol="circle", line=dict(color="#0B0F17", width=2)),
        )
    )

    # Selected Target Date Marker + Price Annotation
    target_plot_dt = pd.to_datetime(f_eff_date)
    fig_future.add_trace(
        go.Scatter(
            x=[target_plot_dt],
            y=[f_pred_price],
            mode="markers+text",
            name=f"Target Date Forecast ({custom_forecast['selected_date']}: {currency_sym}{f_pred_price:,.2f})",
            text=[f"  {custom_forecast['selected_date']}: {currency_sym}{f_pred_price:,.2f}"],
            textposition="top left",
            textfont=dict(color="#34D399" if f_abs_change >= 0 else "#F87171", size=13),
            marker=dict(
                color="#34D399" if f_abs_change >= 0 else "#F87171",
                size=14,
                symbol="star",
                line=dict(color="#FFFFFF", width=1.5),
            ),
        )
    )

    # Vertical dashed divider line separating Historical and Predicted regions
    fig_future.add_vline(
        x=pd.to_datetime(latest_date).timestamp() * 1000,
        line_width=1.5,
        line_dash="dot",
        line_color="rgba(212, 175, 55, 0.55)",
        annotation_text="Forecast Start",
        annotation_position="top left",
    )

    fig_future.update_layout(
        title=(
            f"Historical vs. Forecasted Gold Price — Target: {custom_forecast['selected_date']} "
            f"({f_cal_days} Calendar Days / {f_trad_days} Trading Sessions Ahead)"
        ),
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

    # Step-by-Step Recursive Trajectory & Methodology Explanation
    tab_traj, tab_method = st.tabs([
        f"📋 Step-by-Step Forecast Table ({f_trad_days} Trading Sessions)",
        "🧠 How Recursive Multi-Step Forecasting Works & Its Limitations",
    ])

    with tab_traj:
        st.caption(
            f"Showing all {f_trad_days} sequential trading-session predictions from `{latest_date}` "
            f"to `{f_eff_date}` (for selected target date `{custom_forecast['selected_date']}`)."
        )
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
        st.dataframe(
            traj_display.style.format({
                f"Predicted Price ({short_unit})": "{:,.2f}",
                f"Diff vs {latest_date} ({currency_sym})": "{:+,.2f}",
                "Pct Change (%)": "{:+.2f}%",
                "68% Lower": "{:,.2f}",
                "68% Upper": "{:,.2f}",
                "95% Lower": "{:,.2f}",
                "95% Upper": "{:,.2f}",
            }),
            width="stretch",
            hide_index=True,
        )

    with tab_method:
        st.markdown(
            r"""
            #### 1. Why We Cannot Use the Same 1-Day Prediction for Every Future Date
            A 1-step supervised regression model is trained to predict **tomorrow's** trading session ($P_{t+1}$) from **today's** known lags ($P_t, P_{t-1}, \dots, P_{t-7}$). If you asked for a date 30 or 180 days in the future and simply returned $\hat{P}_{t+1}$, every future date would show the exact same number, ignoring the multi-day trajectory of the model.

            #### 2. How Recursive Multi-Step Forecasting Works
            To predict $H$ trading sessions into the future without using any future actual prices (which are unknown in real life):
            - **Step $h = 1$ (Next Trading Session):** We feed the latest actual historical prices (`2026-01-23` and prior sessions) into the trained model to predict $\hat{P}_{t+1}$.
            - **Step $h = 2$ (Two Trading Sessions Ahead):** We append $\hat{P}_{t+1}$ to our chronological price history. Now $\hat{P}_{t+1}$ becomes `Current_Close`, the actual price on `2026-01-23` shifts into `Lag_1`, and we recompute `MA_7`, `MA_14`, `Pct_Change`, and `Volatility_7` to predict $\hat{P}_{t+2}$.
            - **Step $h = 3 \dots H$:** We repeat this step-by-step across each weekday business session until we reach the user's selected target date.

            #### 3. Key Limitations of Recursive Forecasting
            1. **Error Accumulation (Compounding Drift):** Any small error in Step 1 is fed back as an input feature in Step 2, Step 3, and so on. By Step 14, all 14 prices inside `MA_14` are model predictions rather than observed market prices.
            2. **Variance Attenuation:** Point forecasts smooth out daily random shocks, causing rolling volatility (`Volatility_7`) to decay over long horizons.
            3. **Unforeseen Macroeconomic Events:** No time-series model trained on historical prices can foresee future interest rate decisions, wars, inflation reports, or currency regime changes. Hence, uncertainty cones ($\pm \text{RMSE} \times \sqrt{h}$) widen steadily with the square root of time.
            """
        )


# ===========================================================================
# SECTION D: MODEL COMPARISON
# ===========================================================================
elif nav_section == "D. Model Comparison":
    st.markdown(
        """
        <div class="gold-banner">
            <h1>🏆 Supervised Regression Model Comparison</h1>
            <p>
                Chronological validation performance of all 4 Supervised Machine Learning algorithms
                compared against the Persistence Baseline.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"**Chronological Split Summary:** Training Set: `{split_info['train_start_date']}` to "
        f"`{split_info['train_end_date']}` (**{split_info['train_rows']:,} sessions**, 80%) | "
        f"Validation/Test Set: `{split_info['test_start_date']}` to `{split_info['test_end_date']}` "
        f"(**{split_info['test_rows']:,} sessions**, 20%, unshuffled)."
    )

    # Highlight Best Model Banner
    st.success(
        f"**🥇 Best-Performing Machine Learning Model (Chronological Validation): `{best_model_name}`** — "
        f"Selected based on lowest Test RMSE (**{currency_sym}{metadata['best_model_metrics']['RMSE']:.4f}**) "
        f"and highest Test R² (**{metadata['best_model_metrics']['R2']:.4f}**)."
    )

    # Format Comparison Table
    display_metrics = metrics_df.copy()
    display_metrics["Status"] = display_metrics["Model"].apply(
        lambda m: "🏆 Best ML Model" if m == best_model_name else ("📏 Benchmark" if "Baseline" in m else "Evaluated")
    )
    display_cols = ["Status", "Model", "Type", "MAE", "MSE", "RMSE", "R2", "Train_RMSE", "Train_R2"]

    st.subheader("📋 Validation Metrics Table (Test Split vs. Training Split)")
    st.dataframe(
        display_metrics[display_cols].style.format({
            "MAE": "{:.4f}",
            "MSE": "{:.4f}",
            "RMSE": "{:.4f}",
            "R2": "{:.4f}",
            "Train_RMSE": "{:.4f}",
            "Train_R2": "{:.4f}",
        }),
        width="stretch",
        hide_index=True,
    )

    # Interactive Actual vs Predicted Chart on Chronological Test Set
    st.subheader("📉 Actual vs. Predicted Gold Prices (Chronological Test Set)")
    selected_traces = st.multiselect(
        "Select Models to Plot Against Actual Test Prices:",
        options=[
            "Linear Regression",
            "Persistence Baseline",
            "Random Forest Regression",
            "Gradient Boosting Regression",
            "Decision Tree Regression",
        ],
        default=["Linear Regression", "Persistence Baseline", "Random Forest Regression"],
    )

    fig_comp = go.Figure()
    fig_comp.add_trace(
        go.Scatter(
            x=test_results_df["Target_Date"],
            y=test_results_df["Actual_Next_Close"],
            mode="lines",
            name="Actual Gold Price (Ground Truth)",
            line=dict(color="#F5D061", width=3),
        )
    )

    trace_styles = {
        "Linear Regression": dict(color="#38BDF8", width=2, dash="dot"),
        "Persistence Baseline": dict(color="#94A3B8", width=1.6, dash="dash"),
        "Random Forest Regression": dict(color="#34D399", width=2),
        "Gradient Boosting Regression": dict(color="#A78BFA", width=2),
        "Decision Tree Regression": dict(color="#F87171", width=1.8),
    }

    for trace_name in selected_traces:
        if trace_name in test_results_df.columns:
            fig_comp.add_trace(
                go.Scatter(
                    x=test_results_df["Target_Date"],
                    y=test_results_df[trace_name],
                    mode="lines",
                    name=f"Predicted — {trace_name}",
                    line=trace_styles.get(trace_name, dict(width=2)),
                )
            )

    fig_comp.update_layout(
        title=f"Chronological Validation: Actual vs. Predicted Prices ({split_info['test_start_date']} to {split_info['test_end_date']})",
        template="plotly_dark",
        paper_bgcolor="#141B26",
        plot_bgcolor="#141B26",
        height=480,
        xaxis_title="Target Trading Session Date",
        yaxis_title=f"Gold Price ({unit_label})",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    st.plotly_chart(fig_comp, width="stretch")

    # Bar chart of Test MAE & RMSE
    bcol1, bcol2 = st.columns(2)
    with bcol1:
        fig_mae = px.bar(
            metrics_df,
            x="Model",
            y="MAE",
            color="Type",
            text_auto=".3f",
            title=f"Mean Absolute Error (MAE in {short_unit}) — Lower is Better",
            color_discrete_map={"Machine Learning": "#D4AF37", "Baseline": "#64748B"},
        )
        fig_mae.update_layout(template="plotly_dark", paper_bgcolor="#141B26", plot_bgcolor="#141B26", height=360)
        st.plotly_chart(fig_mae, width="stretch")

    with bcol2:
        fig_rmse = px.bar(
            metrics_df,
            x="Model",
            y="RMSE",
            color="Type",
            text_auto=".3f",
            title=f"Root Mean Squared Error (RMSE in {short_unit}) — Lower is Better",
            color_discrete_map={"Machine Learning": "#F59E0B", "Baseline": "#64748B"},
        )
        fig_rmse.update_layout(template="plotly_dark", paper_bgcolor="#141B26", plot_bgcolor="#141B26", height=360)
        st.plotly_chart(fig_rmse, width="stretch")


# ===========================================================================
# SECTION E: FEATURE IMPORTANCE
# ===========================================================================
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

    st.info(
        "**⚠️ Crucial Scientific Note — Correlation vs. Causation:**\n\n"
        "Feature importance in tree-based models (`Decision Tree`, `Random Forest`, `Gradient Boosting`) "
        "measures how much each historical feature reduces prediction error (impurity) when splitting nodes during training. "
        "**High feature importance is NOT proof of economic causation.** Because daily gold prices are strongly "
        "autocorrelated (today's price starts near yesterday's close), recent price lags (`Current_Close`, `Lag_1`, `MA_7`) "
        "naturally dominate split decisions."
    )

    tree_model_names = list(feature_importances.keys())
    selected_tree_model = st.selectbox(
        "Select Tree-Based Regression Algorithm:",
        options=tree_model_names,
        index=tree_model_names.index("Random Forest Regression") if "Random Forest Regression" in tree_model_names else 0,
    )

    imp_df = feature_importances[selected_tree_model]

    icol1, icol2 = st.columns([1.35, 1.0])
    with icol1:
        fig_imp = px.bar(
            imp_df.sort_values("Importance_Pct", ascending=True),
            x="Importance_Pct",
            y="Feature",
            orientation="h",
            text_auto=".2f",
            title=f"Relative Feature Importance (%) — {selected_tree_model}",
            labels={"Importance_Pct": "Importance Share (%)", "Feature": "Historical Feature"},
            color="Importance_Pct",
            color_continuous_scale=["#334155", "#D4AF37", "#FDE047"],
        )
        fig_imp.update_layout(
            template="plotly_dark",
            paper_bgcolor="#141B26",
            plot_bgcolor="#141B26",
            height=430,
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_imp, width="stretch")

    with icol2:
        st.subheader("📋 Feature Contribution Breakdown")
        st.dataframe(
            imp_df[["Feature", "Importance_Pct", "Description"]].rename(
                columns={"Importance_Pct": "Importance (%)"}
            ).style.format({"Importance (%)": "{:.2f}%"}),
            width="stretch",
            hide_index=True,
        )

    top_feat = imp_df.iloc[0]["Feature"]
    top_pct = imp_df.iloc[0]["Importance_Pct"]
    second_feat = imp_df.iloc[1]["Feature"]
    second_pct = imp_df.iloc[1]["Importance_Pct"]

    st.markdown(
        f"""
        <div class="section-card">
            <h4 style="color:#F5D061; margin-top:0;">🔍 How to Interpret These Results</h4>
            <p>
                In <b>{selected_tree_model}</b>, the top contributor is <code>{top_feat}</code>
                (accounting for <b>{top_pct:.2f}%</b> of total impurity reduction), followed by
                <code>{second_feat}</code> (<b>{second_pct:.2f}%</b>).
            </p>
            <ul style="margin-bottom:0; line-height:1.6;">
                <li><b>Why Recent Price Level Dominates:</b> Financial asset prices follow a near-random-walk (martingale) process over 1-day horizons, where the most recent closing price (<code>Current_Close</code>) anchors the level of the next session's price.</li>
                <li><b>Role of Moving Averages & Lags (<code>MA_7</code>, <code>MA_14</code>, <code>Lag_1..7</code>):</b> Multi-day lags and moving averages capture short-term momentum and local trend smoothing across collinear price features.</li>
                <li><b>Role of Percentage Change & Volatility:</b> Return features capture short-term rate-of-change adjustments around the current price level.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ===========================================================================
# SECTION F: ABOUT PROJECT
# ===========================================================================
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
        """
        ### 1. What is Supervised Machine Learning?
        **Supervised Learning** is a branch of Machine Learning where an algorithm learns a mapping function $f(X) \\rightarrow y$ from labeled historical examples:
        - **Input Features ($X$):** Historical information known at prediction time (e.g., today's closing gold price, lag prices from 1, 2, 3, and 7 trading sessions ago, and 7/14-session moving averages).
        - **Target Label ($y$):** The ground-truth outcome we want to predict — here, the **next available trading session's closing gold price** ($P_{t+1}$).

        During training, the model sees thousands of historical pairs $(X_t, y_t)$ from `2016–2024` and adjusts its internal parameters to minimize prediction error.

        ---

        ### 2. What is Regression?
        In Supervised Learning, problems are divided into two main types based on the target variable $y$:
        - **Classification:** Predicting a discrete category (e.g., *Spam vs. Not Spam*).
        - **Regression:** Predicting a **continuous numerical value** (e.g., predicting that the next trading session's gold price will be **$456.79**). Because gold price is a continuous real-valued number, Gold Price Prediction is a **Regression** problem.

        ---

        ### 3. How Features and Target Variables Are Created (Lag Features)
        Raw time-series data starts as a sequence of dates and closing prices. To apply Supervised Regression without **data leakage**, we transform each trading day $t$ into a feature row using only past and current observations ($\\le t$):
        - **`Current_Close` ($P_t$):** Closing price on session $t$.
        - **`Lag_1` ($P_{t-1}$):** Closing price 1 trading session before $t$ (`Close.shift(1)`).
        - **`Lag_2` ($P_{t-2}$):** Closing price 2 trading sessions before $t$ (`Close.shift(2)`).
        - **`Lag_3` ($P_{t-3}$):** Closing price 3 trading sessions before $t$ (`Close.shift(3)`).
        - **`Lag_7` ($P_{t-7}$):** Closing price 7 trading sessions before $t$ (`Close.shift(7)`).
        - **`MA_7` & `MA_14`:** Average closing price over the last 7 and 14 trading sessions ending on day $t$.
        - **`Pct_Change`:** Percentage change from session $t-1$ to session $t$: $\\frac{P_t - P_{t-1}}{P_{t-1}} \\times 100$.
        - **`Target_Next_Close` ($P_{t+1}$):** Next trading session's closing price (`Close.shift(-1)`).

        ---

        ### 4. Multi-Step Recursive Forecasting for Custom Future Dates
        When predicting a future date more than 1 trading session ahead (up to 365 calendar days):
        1. **Step 1 (Next Trading Session):** Uses the actual historical observations up to `2026-01-23` to predict $\\hat{P}_{t+1}$.
        2. **Step $h > 1$ (Recursive Rollout):** For each subsequent estimated weekday trading session, the previously predicted price $\\hat{P}_{t+h-1}$ is appended to the rolling price history. All lag features (`Lag_1..7`), moving averages (`MA_7`, `MA_14`), and return features are recomputed from this updated series to predict $\\hat{P}_{t+h}$.
        3. **Uncertainty Scaling:** Because recursive errors accumulate over time, the forecast uncertainty standard deviation scales approximately with the square root of the number of trading steps ($\\sigma_h = \\text{RMSE}_1 \\times \\sqrt{h}$).

        ---

        ### 5. Why Chronological Splitting & Persistence Baseline Are Essential
        - **Never Shuffle Time Series:** Randomly shuffling dates would place days from `2025` into the training set and days from `2018` into the test set, allowing the model to "peek into the future" (**Data Leakage**). We strictly train on the oldest 80% (`2016–2024`) and validate on the newest 20% (`2024–2026`).
        - **Persistence Baseline ($\\hat{y}_{t+1} = P_t$):** In daily financial markets, today's price is already a strong predictor of tomorrow's price. Comparing ML models against the Persistence Baseline ensures we evaluate honestly rather than celebrating artificially high $R^2$ scores.
        """
    )
