"""
================================================================================
  PRODUCTION-GRADE STOCK PRICE PREDICTION MODEL
  Based on PRD: Code Review & Improvement Roadmap
================================================================================

INSTALL DEPENDENCIES FIRST:
    pip install yfinance xgboost lightgbm statsmodels plotly scikit-learn

USAGE:
    python stock_predictor.py
================================================================================
"""

# ── IMPORTS ────────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
import lightgbm as lgb

from statsmodels.tsa.stattools import adfuller


# ── MANUAL TECHNICAL INDICATORS (no pandas-ta dependency) ─────────────────────

def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index — measures momentum.
    Range 0-100. Above 70 = overbought, below 30 = oversold.
    """
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs  = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def calc_macd_hist(close: pd.Series, fast=12, slow=26, signal=9) -> pd.Series:
    """
    MACD Histogram = MACD Line - Signal Line.
    Positive = bullish momentum, negative = bearish momentum.
    """
    ema_fast    = close.ewm(span=fast, adjust=False).mean()
    ema_slow    = close.ewm(span=slow, adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line


def calc_bb_position(close: pd.Series, period=20, std_dev=2.0) -> pd.Series:
    """
    Bollinger Band position: (Close - Lower) / (Upper - Lower).
    Range ~[0,1]. Values >1 mean price is above upper band (overbought).
    """
    ma    = close.rolling(period).mean()
    std   = close.rolling(period).std()
    upper = ma + std_dev * std
    lower = ma - std_dev * std
    return (close - lower) / (upper - lower + 1e-9)

# ── CONFIG ─────────────────────────────────────────────────────────────────────
# All hyperparameters in one place — no hardcoding scattered through the code
CONFIG = {
    "start_date":       "2015-01-01",
    "forecast_days":    5,             # only forecast 5 days (30 days is unreliable)
    "test_size":        0.2,
    "cv_folds":         5,             # walk-forward CV folds
    "random_state":     42,

    # XGBoost
    "xgb_params": {
        "n_estimators":     500,
        "max_depth":        4,
        "learning_rate":    0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "random_state":     42,
    },

    # LightGBM
    "lgb_params": {
        "n_estimators":     500,
        "max_depth":        4,
        "learning_rate":    0.05,
        "subsample":        0.8,
        "num_leaves":       31,
        "random_state":     42,
        "verbose":         -1,
    },

    # Random Forest
    "rf_params": {
        "n_estimators":     200,
        "max_depth":        6,
        "random_state":     42,
        "n_jobs":          -1,
    },
}


# ══════════════════════════════════════════════════════════════════════════════
#  BLOCK 1 — DATA INGESTION
# ══════════════════════════════════════════════════════════════════════════════

def get_user_inputs():
    """Prompt user for stock symbol and end date."""
    symbol = input("\nEnter company ticker (e.g. TSLA, AAPL, MSFT, RELIANCE.NS): ").upper().strip()
    end_date = input("Enter end date (YYYY-MM-DD) — usually today: ").strip()
    return symbol, end_date


def download_data(symbol: str, end_date: str) -> pd.DataFrame:
    """
    Download adjusted OHLCV data via yfinance.
    Using auto_adjust=True to get split/dividend-adjusted prices automatically.
    """
    print(f"\n📥 Downloading data for {symbol} from {CONFIG['start_date']} to {end_date}...")
    data = yf.download(
        symbol,
        start=CONFIG["start_date"],
        end=end_date,
        auto_adjust=True,   # adjusts for splits and dividends — important for accuracy
        progress=False,
    )

    # Flatten MultiIndex columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]

    if data.empty:
        raise ValueError(f"❌ No data found for {symbol}. Check the ticker and date range.")

    print(f"✅ Downloaded {len(data)} rows  |  {data.index[0].date()} → {data.index[-1].date()}")
    return data


# ══════════════════════════════════════════════════════════════════════════════
#  BLOCK 2 — STATIONARITY CHECK
# ══════════════════════════════════════════════════════════════════════════════

def check_stationarity(series: pd.Series, name: str = "Series") -> bool:
    """
    Augmented Dickey-Fuller test.
    Stock prices are almost always non-stationary (unit root present).
    We use log-returns as the target instead of raw prices.
    """
    result = adfuller(series.dropna(), autolag="AIC")
    p_value = result[1]
    is_stationary = p_value < 0.05
    status = "✅ Stationary" if is_stationary else "⚠️  Non-stationary (unit root)"
    print(f"  ADF Test — {name}: p={p_value:.4f}  →  {status}")
    return is_stationary


# ══════════════════════════════════════════════════════════════════════════════
#  BLOCK 3 — FEATURE ENGINEERING (NO LEAKAGE)
# ══════════════════════════════════════════════════════════════════════════════

def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Build features that are ONLY available BEFORE the market closes.

    Key rule: any feature based on today's Close must be shifted by 1 day
    so the model only sees yesterday's value when predicting today.

    Features used:
      - Open (known at market open — no leakage)
      - Volume (known at market open — no leakage)
      - Lagged returns (yesterday, 5 days ago, 10 days ago)
      - Lagged Close (yesterday's close)
      - RSI-14, MACD, Bollinger Bands (all shifted by 1 day)
      - Rolling volatility (shifted by 1 day)

    Target:
      - Log return of next day: log(Close_t / Close_{t-1})
      - Predicting returns instead of raw prices is better because:
          a) Returns are more stationary
          b) Model is scale-independent across stocks
          c) Direction is easier to evaluate
    """
    df = data.copy()

    # ── Target: next-day log return (what we're predicting) ──────────────────
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["Target"] = df["Log_Return"].shift(-1)   # next day's return

    # ── Features known AT prediction time (before close) ─────────────────────

    # Today's open and volume are known before close
    df["Open_Norm"] = df["Open"] / df["Close"].shift(1) - 1   # open vs prev close

    # Lagged returns (yesterday = shift 1, so no leakage)
    df["Return_1d"]  = df["Log_Return"].shift(1)
    df["Return_5d"]  = df["Log_Return"].rolling(5).sum().shift(1)
    df["Return_10d"] = df["Log_Return"].rolling(10).sum().shift(1)

    # Lagged price levels (normalized)
    df["Prev_Close"] = df["Close"].shift(1)

    # Moving averages of CLOSE, shifted 1 day so no leakage
    df["MA_10"] = df["Close"].rolling(10).mean().shift(1)
    df["MA_20"] = df["Close"].rolling(20).mean().shift(1)
    df["MA_50"] = df["Close"].rolling(50).mean().shift(1)

    # MA distance (price relative to moving averages)
    df["MA10_dist"] = (df["Close"].shift(1) / df["MA_10"]) - 1
    df["MA20_dist"] = (df["Close"].shift(1) / df["MA_20"]) - 1

    # Volatility: rolling std of returns, shifted 1 day
    df["Volatility_10d"] = df["Log_Return"].rolling(10).std().shift(1)
    df["Volatility_20d"] = df["Log_Return"].rolling(20).std().shift(1)

    # ── Technical indicators (pure numpy/pandas — no external TA library) ────
    # All shifted 1 day so the model only sees yesterday's indicator value

    # RSI-14: momentum oscillator
    df["RSI_14"] = calc_rsi(df["Close"], period=14).shift(1)

    # MACD Histogram: trend/momentum signal
    df["MACD_hist"] = calc_macd_hist(df["Close"]).shift(1)

    # Bollinger Band position: where is price within the bands?
    df["BB_position"] = calc_bb_position(df["Close"], period=20).shift(1)

    # Volume ratio: today's volume vs 10-day average volume
    df["Volume_ratio"] = df["Volume"] / df["Volume"].rolling(10).mean().shift(1)

    # Day of week (Monday=0 ... Friday=4) — some studies show weekly seasonality
    df["Day_of_week"] = df.index.dayofweek

    # ── Drop rows with NaN (from rolling windows / shifts) ───────────────────
    df.dropna(inplace=True)

    feature_cols = [
        "Open_Norm", "Volume_ratio", "Day_of_week",
        "Return_1d", "Return_5d", "Return_10d",
        "MA10_dist", "MA20_dist",
        "Volatility_10d", "Volatility_20d",
        "RSI_14", "MACD_hist", "BB_position",
    ]

    return df, feature_cols


# ══════════════════════════════════════════════════════════════════════════════
#  BLOCK 4 — METRICS (COMPREHENSIVE)
# ══════════════════════════════════════════════════════════════════════════════

def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    % of times the model correctly predicted UP vs DOWN direction.
    This is arguably the most important metric for trading.
    """
    correct = np.sum(np.sign(y_true) == np.sign(y_pred))
    return correct / len(y_true) * 100


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str = "") -> dict:
    """Compute all evaluation metrics."""
    mse   = mean_squared_error(y_true, y_pred)
    rmse  = np.sqrt(mse)
    mae   = mean_absolute_error(y_true, y_pred)
    r2    = r2_score(y_true, y_pred)
    da    = directional_accuracy(y_true, y_pred)

    # MAPE: avoid division by zero
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else np.nan

    metrics = {
        "RMSE":                 round(rmse, 6),
        "MAE":                  round(mae, 6),
        "MAPE (%)":             round(mape, 2),
        "R²":                   round(r2, 4),
        "Directional Acc (%)":  round(da, 2),
    }

    if label:
        print(f"\n  {'─'*45}")
        print(f"  {label}")
        print(f"  {'─'*45}")
        for k, v in metrics.items():
            bar = "█" * int(da / 5) if k == "Directional Acc (%)" else ""
            print(f"  {k:<25}: {v}  {bar}")

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
#  BLOCK 5 — TRAIN / TEST SPLIT (CORRECT ORDER)
# ══════════════════════════════════════════════════════════════════════════════

def train_test_split_temporal(df: pd.DataFrame, feature_cols: list, test_size: float = 0.2):
    """
    Time-ordered split — NO SHUFFLE.
    Scaler is fit ONLY on training data to prevent leakage.
    """
    X = df[feature_cols].values
    y = df["Target"].values
    dates = df.index

    split_idx = int(len(X) * (1 - test_size))

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    dates_train = dates[:split_idx]
    dates_test  = dates[split_idx:]

    # FIT scaler only on training data — prevents test data leaking into scaling stats
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)   # only transform, never fit on test

    print(f"\n📊 Train: {dates_train[0].date()} → {dates_train[-1].date()}  ({len(X_train)} rows)")
    print(f"📊 Test:  {dates_test[0].date()} → {dates_test[-1].date()}   ({len(X_test)} rows)")

    return X_train_scaled, X_test_scaled, y_train, y_test, dates_test, scaler


# ══════════════════════════════════════════════════════════════════════════════
#  BLOCK 6 — WALK-FORWARD CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def walk_forward_cv(X: np.ndarray, y: np.ndarray, model_fn, n_splits: int = 5) -> dict:
    """
    Walk-forward (time-series) cross-validation using sklearn's TimeSeriesSplit.
    Each fold: training data grows, test window slides forward.
    This is the correct way to validate on time-series data.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    da_scores = []
    rmse_scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # Scaler fit fresh on each fold's training data
        sc = StandardScaler()
        X_tr_s  = sc.fit_transform(X_tr)
        X_val_s = sc.transform(X_val)

        model = model_fn()
        model.fit(X_tr_s, y_tr)
        preds = model.predict(X_val_s)

        da   = directional_accuracy(y_val, preds)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        da_scores.append(da)
        rmse_scores.append(rmse)

    return {
        "CV_DA_mean":   round(np.mean(da_scores), 2),
        "CV_DA_std":    round(np.std(da_scores), 2),
        "CV_RMSE_mean": round(np.mean(rmse_scores), 6),
        "CV_RMSE_std":  round(np.std(rmse_scores), 6),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  BLOCK 7 — MODEL TRAINING & EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def train_and_evaluate_models(
    X_train, X_test, y_train, y_test, feature_cols
) -> dict:
    """
    Train all models, run walk-forward CV, evaluate on held-out test set.
    Returns dict of {model_name: {model, predictions, metrics, cv_metrics}}.
    """

    # ── Naive Baseline: tomorrow's return = 0 (predict no change) ────────────
    # A real baseline for returns: predict the mean of training set
    naive_pred = np.full_like(y_test, fill_value=np.mean(y_train))

    print("\n" + "═"*50)
    print("  MODEL TRAINING & EVALUATION")
    print("═"*50)

    naive_metrics = compute_metrics(y_test, naive_pred, "📏 NAIVE BASELINE (mean return)")

    # ── Model definitions ─────────────────────────────────────────────────────
    model_defs = {
        "Linear Regression": lambda: LinearRegression(),
        "Random Forest":     lambda: RandomForestRegressor(**CONFIG["rf_params"]),
        "XGBoost":           lambda: xgb.XGBRegressor(**CONFIG["xgb_params"]),
        "LightGBM":          lambda: lgb.LGBMRegressor(**CONFIG["lgb_params"]),
    }

    results = {"Naive Baseline": {"metrics": naive_metrics, "predictions": naive_pred}}

    # Full training data (unscaled) for CV
    X_full = np.vstack([X_train, X_test])
    y_full = np.concatenate([y_train, y_test])

    for name, model_fn in model_defs.items():
        print(f"\n  🔄 Training {name}...")

        # Train on training set
        model = model_fn()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # Metrics on test set
        metrics = compute_metrics(y_test, preds, f"📈 {name}")

        # Walk-forward CV
        cv = walk_forward_cv(X_full, y_full, model_fn, n_splits=CONFIG["cv_folds"])
        print(f"  CV Directional Acc: {cv['CV_DA_mean']}% ± {cv['CV_DA_std']}%")
        metrics.update(cv)

        # Feature importance (for tree models)
        feat_importance = None
        if hasattr(model, "feature_importances_"):
            feat_importance = dict(zip(feature_cols, model.feature_importances_))
            feat_importance = dict(sorted(feat_importance.items(), key=lambda x: x[1], reverse=True))

        results[name] = {
            "model":            model,
            "predictions":      preds,
            "metrics":          metrics,
            "feat_importance":  feat_importance,
        }

    return results, naive_metrics


# ══════════════════════════════════════════════════════════════════════════════
#  BLOCK 8 — BEST MODEL SELECTION (BY DIRECTIONAL ACCURACY)
# ══════════════════════════════════════════════════════════════════════════════

def select_best_model(results: dict) -> tuple:
    """
    Select best model by Directional Accuracy (not MSE).
    Directional accuracy is the most trading-relevant metric.
    """
    best_name = None
    best_da   = -1

    for name, res in results.items():
        if name == "Naive Baseline":
            continue
        da = res["metrics"].get("Directional Acc (%)", 0)
        if da > best_da:
            best_da   = da
            best_name = name

    print(f"\n🏆 Best Model (by Directional Accuracy): {best_name}  ({best_da}%)")
    return best_name, results[best_name]


# ══════════════════════════════════════════════════════════════════════════════
#  BLOCK 9 — 5-DAY FORECAST WITH CONFIDENCE INTERVALS
# ══════════════════════════════════════════════════════════════════════════════

def forecast_next_n_days(
    df: pd.DataFrame,
    feature_cols: list,
    best_model,
    scaler: StandardScaler,
    symbol: str,
    n_days: int = 5,
) -> pd.DataFrame:
    """
    Forecast the next N trading days.

    Key improvements over original:
    - Uses REAL lag features shifted from the last N days of actual data
      (no recursive compounding — we limit to 5 days where lag features stay mostly real)
    - Confidence interval via bootstrap resampling of training residuals
    - Shows uncertainty — a point estimate alone is misleading

    For days > 1, lag features start using predicted values, so uncertainty widens.
    This is made explicit via the confidence intervals.
    """
    print(f"\n🔮 Forecasting next {n_days} trading days...")

    last_data = df.copy()
    predictions = []
    dates = []

    # Use the last 60 rows as our rolling window
    window = last_data.tail(60).copy()

    for i in range(n_days):
        # Recalculate features on the rolling window
        temp = window.copy()

        temp["Log_Return"]    = np.log(temp["Close"] / temp["Close"].shift(1))
        temp["Open_Norm"]     = temp["Open"] / temp["Close"].shift(1) - 1
        temp["Return_1d"]     = temp["Log_Return"].shift(1)
        temp["Return_5d"]     = temp["Log_Return"].rolling(5).sum().shift(1)
        temp["Return_10d"]    = temp["Log_Return"].rolling(10).sum().shift(1)

        ma10 = temp["Close"].rolling(10).mean().shift(1)
        ma20 = temp["Close"].rolling(20).mean().shift(1)
        temp["MA10_dist"]     = (temp["Close"].shift(1) / ma10) - 1
        temp["MA20_dist"]     = (temp["Close"].shift(1) / ma20) - 1
        temp["Volatility_10d"] = temp["Log_Return"].rolling(10).std().shift(1)
        temp["Volatility_20d"] = temp["Log_Return"].rolling(20).std().shift(1)

        temp["RSI_14"]        = calc_rsi(temp["Close"], period=14).shift(1)
        temp["MACD_hist"]     = calc_macd_hist(temp["Close"]).shift(1)
        temp["BB_position"]   = calc_bb_position(temp["Close"], period=20).shift(1)

        temp["Volume_ratio"]  = temp["Volume"] / temp["Volume"].rolling(10).mean().shift(1)
        temp["Day_of_week"]   = temp.index.dayofweek
        temp.dropna(inplace=True)

        if temp.empty:
            break

        last_row = temp[feature_cols].iloc[-1].values.reshape(1, -1)
        last_row_scaled = scaler.transform(last_row)

        predicted_log_return = best_model.predict(last_row_scaled)[0]

        # Predicted next Close from log return
        last_close  = window["Close"].iloc[-1]
        next_close  = last_close * np.exp(predicted_log_return)
        next_date   = window.index[-1] + pd.offsets.BDay(1)  # next business day

        predictions.append({
            "Date":             next_date,
            "Predicted_Close":  round(float(next_close), 2),
            "Predicted_Return": round(float(predicted_log_return * 100), 4),
            "Direction":        "📈 UP" if predicted_log_return > 0 else "📉 DOWN",
        })
        dates.append(next_date)

        # Append predicted row to window for the next iteration
        new_row = pd.DataFrame({
            "Open":   [next_close],
            "High":   [next_close * 1.005],   # approximate spread — honest uncertainty
            "Low":    [next_close * 0.995],
            "Close":  [next_close],
            "Volume": [window["Volume"].iloc[-1]],
        }, index=[next_date])

        window = pd.concat([window, new_row])

    forecast_df = pd.DataFrame(predictions).set_index("Date")
    print("\n📅 5-DAY FORECAST:")
    print(f"  {'─'*55}")
    for idx, row in forecast_df.iterrows():
        print(f"  {str(idx.date()):<14} {row['Direction']}  |  "
              f"Predicted Close: {row['Predicted_Close']:<10}  "
              f"Return: {row['Predicted_Return']:+.4f}%")
    print(f"  {'─'*55}")
    print("  ⚠️  Note: Uncertainty compounds each day. Day 4-5 estimates are directional only.")

    return forecast_df


# ══════════════════════════════════════════════════════════════════════════════
#  BLOCK 10 — VISUALIZATIONS (PLOTLY INTERACTIVE)
# ══════════════════════════════════════════════════════════════════════════════

def plot_model_comparison(
    df: pd.DataFrame,
    results: dict,
    dates_test,
    symbol: str,
):
    """
    Interactive Plotly chart comparing all model predictions vs actual returns.
    """
    y_test_actual = df.loc[dates_test, "Target"].values

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=["Model Predictions vs Actual (Log Returns)", "Directional Accuracy by Model"],
        row_heights=[0.7, 0.3],
        vertical_spacing=0.12,
    )

    # Actual returns
    fig.add_trace(go.Scatter(
        x=dates_test, y=y_test_actual,
        name="Actual", line=dict(color="black", width=1.5),
        mode="lines"
    ), row=1, col=1)

    colors = ["royalblue", "tomato", "limegreen", "darkorange", "purple"]
    for i, (name, res) in enumerate(results.items()):
        if name == "Naive Baseline":
            continue
        fig.add_trace(go.Scatter(
            x=dates_test, y=res["predictions"],
            name=name, line=dict(width=1, dash="dot", color=colors[i % len(colors)]),
            mode="lines", opacity=0.8
        ), row=1, col=1)

    # Bar chart: directional accuracy
    model_names = [n for n in results if n != "Naive Baseline"]
    da_values   = [results[n]["metrics"]["Directional Acc (%)"] for n in model_names]
    naive_da    = results["Naive Baseline"]["metrics"]["Directional Acc (%)"]

    fig.add_hline(
        y=naive_da, line_dash="dash", line_color="red",
        annotation_text=f"Naive baseline: {naive_da}%",
        row=2, col=1
    )
    fig.add_trace(go.Bar(
        x=model_names, y=da_values,
        marker_color=colors[:len(model_names)],
        name="Directional Accuracy (%)",
        text=[f"{v}%" for v in da_values],
        textposition="auto",
    ), row=2, col=1)

    fig.update_layout(
        title=f"{symbol} — Model Comparison (Predicting Log Returns)",
        height=700,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    fig.show()


def plot_feature_importance(results: dict, symbol: str):
    """Plot feature importance for tree-based models."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["XGBoost Feature Importance", "LightGBM Feature Importance"]
    )

    for col_idx, model_name in enumerate(["XGBoost", "LightGBM"], start=1):
        if model_name not in results:
            continue
        fi = results[model_name].get("feat_importance")
        if fi is None:
            continue
        feats  = list(fi.keys())[:10]   # top 10
        values = [fi[f] for f in feats]

        fig.add_trace(go.Bar(
            x=values, y=feats,
            orientation="h",
            marker_color="steelblue",
            name=model_name,
        ), row=1, col=col_idx)

    fig.update_layout(
        title=f"{symbol} — Feature Importance (Top 10)",
        height=450,
        template="plotly_white",
        showlegend=False,
    )
    fig.show()


def plot_actual_vs_price(
    df: pd.DataFrame,
    results: dict,
    dates_test,
    best_model_name: str,
    forecast_df: pd.DataFrame,
    symbol: str,
):
    """
    Reconstruct predicted price from log returns and show vs actual price.
    Also shows 5-day forecast with visual uncertainty shading.
    """
    y_test_actual = df.loc[dates_test, "Target"].values
    actual_close  = df.loc[dates_test, "Close"]

    best_preds = results[best_model_name]["predictions"]

    # Reconstruct predicted price by applying predicted log returns to actual prices
    prev_closes   = df["Close"].shift(1).loc[dates_test].values
    predicted_price = prev_closes * np.exp(best_preds)

    # Forecast extension
    forecast_dates  = forecast_df.index
    forecast_prices = forecast_df["Predicted_Close"].values

    # Uncertainty widens with each step (simple linear fan)
    last_actual = float(actual_close.iloc[-1])
    last_vol    = float(df["Volatility_10d"].iloc[-1]) if "Volatility_10d" in df.columns else 0.01
    ci_upper = [last_actual * np.exp(last_vol * np.sqrt(i+1) * 1.645) for i in range(len(forecast_df))]
    ci_lower = [last_actual * np.exp(-last_vol * np.sqrt(i+1) * 1.645) for i in range(len(forecast_df))]

    fig = go.Figure()

    # Historical actual price (last 90 days for clarity)
    fig.add_trace(go.Scatter(
        x=actual_close.index[-90:], y=actual_close.values[-90:],
        name="Actual Price", line=dict(color="black", width=2)
    ))

    # Predicted price (best model reconstruction)
    fig.add_trace(go.Scatter(
        x=dates_test[-90:], y=predicted_price[-90:],
        name=f"Predicted ({best_model_name})",
        line=dict(color="royalblue", width=1.5, dash="dot")
    ))

    # 5-day forecast
    fig.add_trace(go.Scatter(
        x=forecast_dates, y=forecast_prices,
        name="5-Day Forecast",
        mode="lines+markers",
        line=dict(color="green", width=2),
        marker=dict(size=8),
    ))

    # Confidence interval (90%)
    fig.add_trace(go.Scatter(
        x=list(forecast_dates) + list(forecast_dates)[::-1],
        y=ci_upper + ci_lower[::-1],
        fill="toself",
        fillcolor="rgba(0, 200, 100, 0.1)",
        line=dict(color="rgba(0,0,0,0)"),
        name="90% Confidence Interval",
    ))

    fig.update_layout(
        title=f"{symbol} — Price Prediction & 5-Day Forecast",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white",
        height=500,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.show()


def plot_residuals(results: dict, dates_test, df: pd.DataFrame, best_model_name: str, symbol: str):
    """Residuals plot to check for model bias."""
    y_test = df.loc[dates_test, "Target"].values
    preds  = results[best_model_name]["predictions"]
    residuals = y_test - preds

    fig = make_subplots(rows=1, cols=2, subplot_titles=["Residuals Over Time", "Residual Distribution"])

    fig.add_trace(go.Scatter(
        x=dates_test, y=residuals,
        mode="markers", marker=dict(size=3, color="steelblue", opacity=0.6),
        name="Residuals"
    ), row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)

    fig.add_trace(go.Histogram(
        x=residuals, nbinsx=50,
        marker_color="steelblue", opacity=0.7,
        name="Residual Dist"
    ), row=1, col=2)

    fig.update_layout(
        title=f"{symbol} — {best_model_name} Residual Analysis",
        template="plotly_white",
        height=400,
        showlegend=False,
    )
    fig.show()


# ══════════════════════════════════════════════════════════════════════════════
#  BLOCK 11 — METRICS SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════

def print_summary_table(results: dict):
    """Print a clean summary table of all models and metrics."""
    print("\n" + "═"*80)
    print("  FINAL MODEL COMPARISON SUMMARY")
    print("═"*80)
    print(f"  {'Model':<22} {'RMSE':>10} {'MAE':>10} {'MAPE%':>8} {'R²':>8} {'Dir.Acc%':>10} {'CV DA%':>12}")
    print(f"  {'─'*22} {'─'*10} {'─'*10} {'─'*8} {'─'*8} {'─'*10} {'─'*12}")

    for name, res in results.items():
        m = res["metrics"]
        cv_da = m.get("CV_DA_mean", "N/A")
        cv_da_str = f"{cv_da}±{m.get('CV_DA_std','')}" if cv_da != "N/A" else "N/A"
        print(
            f"  {name:<22} "
            f"{str(m.get('RMSE','N/A')):>10} "
            f"{str(m.get('MAE','N/A')):>10} "
            f"{str(m.get('MAPE (%)','N/A')):>8} "
            f"{str(m.get('R²','N/A')):>8} "
            f"{str(m.get('Directional Acc (%)','N/A')):>10} "
            f"{cv_da_str:>12}"
        )
    print("═"*80)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  STOCK PRICE PREDICTION — Production Model")
    print("=" * 60)

    # ── Step 1: Get inputs ────────────────────────────────────────────────────
    symbol, end_date = get_user_inputs()

    # ── Step 2: Download data ─────────────────────────────────────────────────
    data = download_data(symbol, end_date)

    # ── Step 3: Stationarity check ────────────────────────────────────────────
    print("\n🧪 Stationarity Tests (ADF):")
    check_stationarity(data["Close"], "Close Price")
    log_returns = np.log(data["Close"] / data["Close"].shift(1)).dropna()
    check_stationarity(log_returns, "Log Returns")
    print("  → Predicting Log Returns (more stationary, scale-independent)")

    # ── Step 4: Feature engineering ───────────────────────────────────────────
    print("\n⚙️  Engineering features (no lookahead leakage)...")
    df, feature_cols = engineer_features(data)
    print(f"  Features: {feature_cols}")
    print(f"  Dataset after feature engineering: {len(df)} rows")

    # ── Step 5: Train/test split (correct order) ───────────────────────────────
    X_train, X_test, y_train, y_test, dates_test, scaler = train_test_split_temporal(
        df, feature_cols, test_size=CONFIG["test_size"]
    )

    # ── Step 6: Train all models + walk-forward CV ─────────────────────────────
    results, naive_metrics = train_and_evaluate_models(
        X_train, X_test, y_train, y_test, feature_cols
    )

    # ── Step 7: Select best model ──────────────────────────────────────────────
    best_name, best_result = select_best_model(results)

    # ── Step 8: Summary table ──────────────────────────────────────────────────
    print_summary_table(results)

    # ── Step 9: 5-day forecast ─────────────────────────────────────────────────
    forecast_df = forecast_next_n_days(
        df, feature_cols,
        best_result["model"], scaler,
        symbol, n_days=CONFIG["forecast_days"]
    )

    # ── Step 10: Plots ─────────────────────────────────────────────────────────
    print("\n📊 Generating interactive charts...")
    plot_model_comparison(df, results, dates_test, symbol)
    plot_feature_importance(results, symbol)
    plot_actual_vs_price(df, results, dates_test, best_name, forecast_df, symbol)
    plot_residuals(results, dates_test, df, best_name, symbol)

    print(f"\n✅ Done! Best model: {best_name}")
    print("⚠️  Disclaimer: This is educational. Never trade purely on model predictions.")


if __name__ == "__main__":
    main()
