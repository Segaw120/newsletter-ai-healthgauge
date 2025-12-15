import streamlit as st
import pandas as pd
import numpy as np
import datetime
import time
import os
import logging
import threading
import plotly.graph_objects as go
import plotly.express as px
from sodapy import Socrata
from yahooquery import Ticker
from huggingface_hub import InferenceClient
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from scipy import stats

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Constants and Setup ---
SODAPY_APP_TOKEN = "PP3ezxaUTiGforvvbBUGzwRx7"
client = Socrata("publicreporting.cftc.gov", SODAPY_APP_TOKEN, timeout=60)

# Initialize HF client once, expects HF_TOKEN in environment
hf_client = InferenceClient(provider="cerebras", api_key=os.getenv("HF_TOKEN"))

# --- Asset mapping (COT market names -> Yahoo futures tickers) ---
assets = {
    "GOLD - COMMODITY EXCHANGE INC.": "GC=F",
    "SILVER - COMMODITY EXCHANGE INC.": "SI=F",
    "PLATINUM - NEW YORK MERCANTILE EXCHANGE": "PL=F",
    "PALLADIUM - NEW YORK MERCANTILE EXCHANGE": "PA=F",
    "WTI CRUDE OIL FINANCIAL - NEW YORK MERCANTILE EXCHANGE": "CL=F",
    "COPPER - COMMODITY EXCHANGE INC.": "HG=F",
    "AUSTRALIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE": "6A=F",
    "BRITISH POUND - CHICAGO MERCANTILE EXCHANGE": "6B=F",
    "CANADIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE": "6C=F",
    "EURO FX - CHICAGO MERCANTILE EXCHANGE": "6E=F",
    "JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE": "6J=F",
    "SWISS FRANC - CHICAGO MERCANTILE EXCHANGE": "6S=F",
    "BITCOIN - CHICAGO MERCANTILE EXCHANGE": "BTC=F",
    "MICRO BITCOIN - CHICAGO MERCANTILE EXCHANGE": "MBT=F",
    "MICRO ETHER - CHICAGO MERCANTILE EXCHANGE": "ETH=F",
    "E-MINI S&P FINANCIAL INDEX - CHICAGO MERCANTILE EXCHANGE": "ES=F",
    "DOW JONES U.S. REAL ESTATE IDX - CHICAGO BOARD OF TRADE": "DJR",
    "NATURAL GAS - NEW YORK MERCANTILE EXCHANGE": "NG=F",
    "CORN - CHICAGO BOARD OF TRADE": "ZC=F",
    "SOYBEANS - CHICAGO BOARD OF TRADE": "ZS=F",
}

# --- Fetch COT Data ---
def fetch_cot_data(market_name: str, max_attempts: int = 3) -> pd.DataFrame:
    logger.info(f"Fetching COT data for {market_name}")
    where_clause = f'market_and_exchange_names="{market_name}"'
    attempt = 0
    while attempt < max_attempts:
        try:
            results = client.get(
                "6dca-aqww",
                where=where_clause,
                order="report_date_as_yyyy_mm_dd DESC",
                limit=1500,
            )
            if results:
                df = pd.DataFrame.from_records(results)
                df["report_date"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"])
                df["open_interest_all"] = pd.to_numeric(df["open_interest_all"], errors="coerce")
                try:
                    df["commercial_long"] = pd.to_numeric(df["commercial_long_all"], errors="coerce")
                    df["commercial_short"] = pd.to_numeric(df["commercial_short_all"], errors="coerce")
                    df["non_commercial_long"] = pd.to_numeric(df["non_commercial_long_all"], errors="coerce")
                    df["non_commercial_short"] = pd.to_numeric(df["non_commercial_short_all"], errors="coerce")
                    
                    df["commercial_net"] = df["commercial_long"] - df["commercial_short"]
                    df["non_commercial_net"] = df["non_commercial_long"] - df["non_commercial_short"]
                    
                    # Calculate commercial position index (0-100)
                    df["commercial_position_pct"] = (df["commercial_long"] / 
                                                    (df["commercial_long"] + df["commercial_short"])) * 100
                    
                    # Calculate non-commercial position index (0-100)
                    df["non_commercial_position_pct"] = (df["non_commercial_long"] / 
                                                        (df["non_commercial_long"] + df["non_commercial_short"])) * 100
                    
                    # Calculate z-scores for positioning extremes
                    df["commercial_net_zscore"] = (df["commercial_net"] - 
                                                 df["commercial_net"].rolling(52).mean()) / df["commercial_net"].rolling(52).std()
                    df["non_commercial_net_zscore"] = (df["non_commercial_net"] - 
                                                     df["non_commercial_net"].rolling(52).mean()) / df["non_commercial_net"].rolling(52).std()
                    
                except KeyError:
                    df["commercial_net"] = 0.0
                    df["non_commercial_net"] = 0.0
                    df["commercial_position_pct"] = 50.0
                    df["non_commercial_position_pct"] = 50.0
                    df["commercial_net_zscore"] = 0.0
                    df["non_commercial_net_zscore"] = 0.0
                
                return df.sort_values("report_date")
            else:
                logger.warning(f"No COT data for {market_name}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching COT data for {market_name}: {e}")
        attempt += 1
        time.sleep(2)
    logger.error(f"Failed fetching COT data for {market_name} after {max_attempts} attempts.")
    return pd.DataFrame()

# --- Fetch Price Data (yahooquery) ---
def fetch_yahooquery_data(ticker: str, start_date: str, end_date: str, max_attempts: int = 3) -> pd.DataFrame:
    logger.info(f"Fetching Yahoo data for {ticker} from {start_date} to {end_date}")
    attempt = 0
    while attempt < max_attempts:
        try:
            t = Ticker(ticker)
            hist = t.history(start=start_date, end=end_date, interval="1d")
            if hist.empty:
                logger.warning(f"No price data for {ticker}")
                return pd.DataFrame()
            if isinstance(hist.index, pd.MultiIndex):
                hist = hist.loc[ticker]
            hist = hist.reset_index()
            hist["date"] = pd.to_datetime(hist["date"])
            
            # Calculate technical indicators
            hist = calculate_technical_indicators(hist)
            
            return hist.sort_values("date")
        except Exception as e:
            logger.error(f"Error fetching Yahoo data for {ticker}: {e}")
        attempt += 1
        time.sleep(2)
    logger.error(f"Failed fetching Yahoo data for {ticker} after {max_attempts} attempts.")
    return pd.DataFrame()

# --- Calculate Technical Indicators ---
def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    
    close_col = "close" if "close" in df.columns else ("Close" if "Close" in df.columns else None)
    high_col = "high" if "high" in df.columns else ("High" if "High" in df.columns else None)
    low_col = "low" if "low" in df.columns else ("Low" if "Low" in df.columns else None)
    
    if not all([close_col, high_col, low_col]):
        return df
    
    # Calculate RVOL
    vol_col = "volume" if "volume" in df.columns else ("Volume" if "Volume" in df.columns else None)
    if vol_col is not None:
        rolling_avg = df[vol_col].rolling(20).mean()
        df["rvol"] = df[vol_col] / rolling_avg
    
    # Calculate RSI
    delta = df[close_col].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))
    
    # Calculate Moving Averages
    df["sma20"] = df[close_col].rolling(20).mean()
    df["sma50"] = df[close_col].rolling(50).mean()
    df["sma200"] = df[close_col].rolling(200).mean()
    
    # Calculate Bollinger Bands
    df["bb_middle"] = df[close_col].rolling(20).mean()
    df["bb_std"] = df[close_col].rolling(20).std()
    df["bb_upper"] = df["bb_middle"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_middle"] - 2 * df["bb_std"]
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
    
    # Calculate ATR
    tr1 = df[high_col] - df[low_col]
    tr2 = abs(df[high_col] - df[close_col].shift())
    tr3 = abs(df[low_col] - df[close_col].shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()
    
    # Volatility
    df["volatility"] = df[close_col].pct_change().rolling(20).std() * np.sqrt(252) * 100
    
    # Calculate distance from 52-week high/low
    df["52w_high"] = df[close_col].rolling(252).max()
    df["52w_low"] = df[close_col].rolling(252).min()
    df["pct_from_52w_high"] = (df[close_col] / df["52w_high"] - 1) * 100
    df["pct_from_52w_low"] = (df[close_col] / df["52w_low"] - 1) * 100
    
    return df

# --- Merge COT and Price Data ---
def merge_cot_price(cot_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    if cot_df.empty or price_df.empty:
        return pd.DataFrame()
    
    cot_columns = ["report_date", "open_interest_all", "commercial_net", "non_commercial_net", 
                  "commercial_position_pct", "non_commercial_position_pct", 
                  "commercial_net_zscore", "non_commercial_net_zscore"]
    
    # Ensure all required columns exist
    for col in cot_columns:
        if col not in cot_df.columns:
            cot_df[col] = np.nan
    
    cot_df_small = cot_df[cot_columns].copy()
    cot_df_small.rename(columns={"report_date": "date"}, inplace=True)
    cot_df_small["date"] = pd.to_datetime(cot_df_small["date"])
    
    price_df = price_df.copy()
    price_df["date"] = pd.to_datetime(price_df["date"])
    price_df = price_df.sort_values("date")
    cot_df_small = cot_df_small.sort_values("date")
    
    full_dates = pd.DataFrame({"date": pd.date_range(price_df["date"].min(), price_df["date"].max())})
    cot_df_filled = pd.merge_asof(full_dates, cot_df_small, on="date", direction="backward")
    
    merged = pd.merge(price_df, cot_df_filled, on="date", how="left")
    
    # Forward fill COT data (carried forward until next COT report)
    for col in cot_columns[1:]:
        merged[col] = merged[col].ffill()
    
    return merged

# --- Calculate Health Gauge ---
def calculate_health_gauge(merged_df: pd.DataFrame) -> float:
    if merged_df.empty:
        return np.nan
    
    # Get latest data
    latest = merged_df.tail(1).iloc[0]
    recent = merged_df.tail(90).copy()
    
    close_col = "close" if "close" in recent.columns else ("Close" if "Close" in recent.columns else None)
    
    if close_col is None:
        return np.nan
    
    scores = []
    
    # 1. Commercial net position extreme score (25%)
    if "commercial_net_zscore" in latest and not pd.isna(latest["commercial_net_zscore"]):
        # Commercials are smart money - going against the crowd
        # Higher score when commercials are net long against the crowd (negative z-score)
        comm_score = max(0, min(1, 0.5 - latest["commercial_net_zscore"]/4))
        scores.append((comm_score, 0.25))
    
    # 2. Trend alignment score (20%)
    if all(x in recent.columns for x in [close_col, "sma20", "sma50", "sma200"]):
        last_close = latest[close_col]
        trend_signals = [
            last_close > latest["sma20"],  # Above 20-day MA
            latest["sma20"] > latest["sma50"],  # 20-day MA above 50-day MA
            latest["sma50"] > latest["sma200"],  # 50-day MA above 200-day MA
        ]
        trend_score = sum(trend_signals) / len(trend_signals)
        scores.append((trend_score, 0.20))
    
    # 3. Momentum score (15%)
    if "rsi" in recent.columns:
        # RSI score - prefer middle-high range (40-70)
        rsi = latest["rsi"]
        if pd.isna(rsi):
            rsi_score = 0.5
        elif rsi < 30:
            rsi_score = 0.3  # Oversold
        elif rsi > 70:
            rsi_score = 0.7  # Overbought
        else:
            rsi_score = 0.5 + (rsi - 50) / 100  # Linear between 0.5 and 0.7
        scores.append((rsi_score, 0.15))
    
    # 4. Volatility and volume score (15%)
    vol_score = 0.5
    if "bb_width" in recent.columns and "rvol" in recent.columns:
        # Normalize BB width
        bb_width_percentile = stats.percentileofscore(
            recent["bb_width"].dropna(), latest["bb_width"]) / 100
        
        # Higher score for contracting volatility (coiling for move)
        bb_score = 1 - bb_width_percentile
        
        # Volume score - recent relative volume
        rvol_score = min(1.0, latest["rvol"] / 2.0) if not pd.isna(latest["rvol"]) else 0.5
        
        vol_score = 0.7 * bb_score + 0.3 * rvol_score
        scores.append((vol_score, 0.15))
    
    # 5. Distance from 52-week high/low score (15%)
    if "pct_from_52w_high" in recent.columns and "pct_from_52w_low" in recent.columns:
        # Higher score when closer to 52-week high
        high_score = 1 - (abs(latest["pct_from_52w_high"]) / 100)
        high_score = max(0, min(1, high_score))
        
        # Higher score when further from 52-week low
        low_score = min(1, latest["pct_from_52w_low"] / 100)
        low_score = max(0, min(1, low_score))
        
        # Combine with preference to high_score
        dist_score = 0.7 * high_score + 0.3 * low_score
        scores.append((dist_score, 0.15))
    
    # 6. Open interest score (10%)
    if "open_interest_all" in recent.columns:
        oi = recent["open_interest_all"].dropna()
        if not oi.empty:
            oi_pctile = stats.percentileofscore(oi, latest["open_interest_all"]) / 100
            # Prefer higher open interest
            oi_score = oi_pctile
            scores.append((oi_score, 0.10))
    
    # Calculate weighted score
    if not scores:
        return 5.0  # Neutral if no data
    
    weighted_sum = sum(score * weight for score, weight in scores)
    total_weight = sum(weight for _, weight in scores)
    
    # Scale to 0-10
    health_score = (weighted_sum / total_weight) * 10
    
    return float(health_score)

# --- Signal Generation ---
def generate_signals(merged_df: pd.DataFrame) -> dict:
    if merged_df.empty:
        return {"signal": "NEUTRAL", "strength": 0, "reasoning": "Insufficient data"}
    
    close_col = "close" if "close" in merged_df.columns else ("Close" if "Close" in merged_df.columns else None)
    if close_col is None:
        return {"signal": "NEUTRAL", "strength": 0, "reasoning": "Price data missing"}
    
    recent = merged_df.tail(30).copy()
    latest = recent.iloc[-1]
    
    signal_reasons = []
    bullish_points = 0
    bearish_points = 0
    
    # 1. COT commercial position signals
    if "commercial_net_zscore" in latest and not pd.isna(latest["commercial_net_zscore"]):
        if latest["commercial_net_zscore"] < -1.5:
            bullish_points += 2
            signal_reasons.append("Commercials heavily net long (contrarian bullish)")
        elif latest["commercial_net_zscore"] < -0.5:
            bullish_points += 1
            signal_reasons.append("Commercials moderately net long")
        elif latest["commercial_net_zscore"] > 1.5:
            bearish_points += 2
            signal_reasons.append("Commercials heavily net short (contrarian bearish)")
        elif latest["commercial_net_zscore"] > 0.5:
            bearish_points += 1
            signal_reasons.append("Commercials moderately net short")
    
    # 2. Price trend signals
    if all(x in latest for x in [close_col, "sma20", "sma50", "sma200"]):
        if latest[close_col] > latest["sma20"] > latest["sma50"] > latest["sma200"]:
            bullish_points += 2
            signal_reasons.append("Strong uptrend: price above all major MAs")
        elif latest[close_col] > latest["sma50"] and latest["sma50"] > latest["sma200"]:
            bullish_points += 1
            signal_reasons.append("Uptrend: price above 50 and 200-day MAs")
        elif latest[close_col] < latest["sma20"] < latest["sma50"] < latest["sma200"]:
            bearish_points += 2
            signal_reasons.append("Strong downtrend: price below all major MAs")
        elif latest[close_col] < latest["sma50"] and latest["sma50"] < latest["sma200"]:
            bearish_points += 1
            signal_reasons.append("Downtrend: price below 50 and 200-day MAs")
    
    # 3. RSI signals
    if "rsi" in latest and not pd.isna(latest["rsi"]):
        if latest["rsi"] < 30:
            bullish_points += 1
            signal_reasons.append("RSI oversold (below 30)")
        elif latest["rsi"] < 40:
            bullish_points += 0.5
            signal_reasons.append("RSI approaching oversold")
        elif latest["rsi"] > 70:
            bearish_points += 1
            signal_reasons.append("RSI overbought (above 70)")
        elif latest["rsi"] > 60:
            bearish_points += 0.5
            signal_reasons.append("RSI approaching overbought")
    
    # 4. Bollinger Band signals
    if all(x in latest for x in ["bb_upper", "bb_lower", close_col]):
        if latest[close_col] > latest["bb_upper"]:
            bearish_points += 1
            signal_reasons.append("Price above upper Bollinger Band")
        elif latest[close_col] < latest["bb_lower"]:
            bullish_points += 1
            signal_reasons.append("Price below lower Bollinger Band")
    
    # 5. Volume signals
    if "rvol" in latest and not pd.isna(latest["rvol"]):
        if latest["rvol"] > 1.5:
            # Check if volume spike is bullish or bearish
            if close_col and len(recent) > 1:
                price_change = latest[close_col] - recent.iloc[-2][close_col]
                if price_change > 0:
                    bullish_points += 1
                    signal_reasons.append("High volume on price advance")
                elif price_change < 0:
                    bearish_points += 1
                    signal_reasons.append("High volume on price decline")
    
    # 6. 52-week signals
    if "pct_from_52w_high" in latest and "pct_from_52w_low" in latest:
        if latest["pct_from_52w_high"] > -5:
            bullish_points += 1
            signal_reasons.append("Price near 52-week high")
        elif latest["pct_from_52w_low"] < 10:
            bearish_points += 1
            signal_reasons.append("Price near 52-week low")
    
    # Determine overall signal
    net_score = bullish_points - bearish_points
    
    if net_score >= 3:
        signal = "STRONG BUY"
        strength = min(5, int(net_score))
    elif net_score > 0:
        signal = "BUY"
        strength = min(3, int(net_score))
    elif net_score <= -3:
        signal = "STRONG SELL"
        strength = min(5, int(abs(net_score)))
    elif net_score < 0:
        signal = "SELL"
        strength = min(3, int(abs(net_score)))
    else:
        signal = "NEUTRAL"
        strength = 0
    
    # Add health score to reasoning
    if "health_score" in merged_df.columns:
        health_score = merged_df["health_score"].iloc[-1]
        signal_reasons.append(f"Health gauge: {health_score:.2f}/10")
    
    return {
        "signal": signal,
        "strength": strength,
        "reasoning": "; ".join(signal_reasons)
    }

# --- Calculate RVOL (Relative Volume) ---
def calculate_rvol(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    vol_col = "volume" if "volume" in df.columns else ("Volume" if "Volume" in df.columns else None)
    if vol_col is None:
        df["rvol"] = np.nan
        return df
    rolling_avg = df[vol_col].rolling(window).mean()
    df["rvol"] = df[vol_col] / rolling_avg
    return df

# --- Generate Advanced Forecasts ---
def generate_signals(df: pd.DataFrame, cot_df: pd.DataFrame) -> list:
    signals = []
    if df is None or df.empty or cot_df is None or cot_df.empty:
        return signals

    latest = df.iloc[-1]
    cot_latest = cot_df.iloc[-1]

    # Example trading rules
    if (
        latest["rsi"] < 30
        and latest["close"] > latest["sma50"]
        and cot_latest["noncomm_positions_change"] > 0
    ):
        signals.append(
            {
                "signal": "BUY",
                "reason": "RSI oversold, price above SMA50, non-commercial longs increasing",
            }
        )
    elif (
        latest["rsi"] > 70
        and latest["close"] < latest["sma50"]
        and cot_latest["noncomm_positions_change"] < 0
    ):
        signals.append(
            {
                "signal": "SELL",
                "reason": "RSI overbought, price below SMA50, non-commercial longs decreasing",
            }
        )
    else:
        signals.append(
            {
                "signal": "NEUTRAL",
                "reason": "No strong signal detected",
            }
        )

    return signals


# --- Relative Volume Calculation ---
def calculate_rvol(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    if "volume" not in df.columns:
        return pd.Series([np.nan] * len(df), index=df.index)
    avg_volume = df["volume"].rolling(window=lookback).mean()
    return df["volume"] / avg_volume


# --- Trade Setup Generation ---
def generate_trade_setup(signal: str, df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {}
    latest = df.iloc[-1]

    setup = {"signal": signal}
    atr = latest.get("atr", np.nan)

    if signal == "BUY":
        setup["stop_loss"] = latest["close"] - (2 * atr)
        setup["target"] = latest["close"] + (3 * atr)
    elif signal == "SELL":
        setup["stop_loss"] = latest["close"] + (2 * atr)
        setup["target"] = latest["close"] - (3 * atr)
    else:
        setup["stop_loss"] = None
        setup["target"] = None

    return setup


# --- AI Market Newsletter Generation ---
def build_newsletter_prompt(asset: str, signals: list, df: pd.DataFrame, cot_df: pd.DataFrame) -> str:
    if df is None or df.empty or cot_df is None or cot_df.empty:
        return f"No sufficient data available for {asset}."

    latest = df.iloc[-1]
    cot_latest = cot_df.iloc[-1]
    signal_texts = [f"{s['signal']} ({s['reason']})" for s in signals]

    prompt = f"""
    Generate a professional market newsletter for {asset}.

    Key Data:
    - Latest close price: {latest['close']:.2f}
    - RSI: {latest['rsi']:.2f}
    - SMA50: {latest['sma50']:.2f}
    - SMA200: {latest['sma200']:.2f}
    - Bollinger Bands: {latest['bb_low']:.2f} – {latest['bb_high']:.2f}
    - ATR: {latest['atr']:.2f}
    - Relative Volume: {latest.get('rvol', np.nan):.2f}
    - 52w High/Low: {latest['52w_low']:.2f} – {latest['52w_high']:.2f}

    Commitment of Traders:
    - Non-commercial net positions: {cot_latest['noncomm_net']}
    - Commercial net positions: {cot_latest['commercial_net']}

    Signals: {", ".join(signal_texts)}

    Write in a structured, concise, professional style.
    """
    return prompt.strip()


def generate_market_newsletter(prompt: str) -> str:
    try:
        response = hf_client.text_generation(
            model="mistralai/Mistral-7B-Instruct-v0.3",
            inputs=prompt,
            parameters={"max_new_tokens": 500},
        )
        return response.get("generated_text", "").strip()
    except Exception as e:
        logger.error(f"Newsletter generation failed: {e}")
        return "Error generating newsletter."


# --- Visualization ---
def create_asset_chart(df: pd.DataFrame, cot_df: pd.DataFrame, asset_name: str):
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.05,
        subplot_titles=(f"{asset_name} Price & Indicators", "COT Non-Commercial Net Positions")
    )

    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="Price"
    ), row=1, col=1)

    # Moving averages
    fig.add_trace(go.Scatter(x=df.index, y=df["sma50"], mode="lines", name="SMA50"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["sma200"], mode="lines", name="SMA200"), row=1, col=1)

    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=df["bb_high"], mode="lines", line=dict(width=1), name="BB High"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["bb_low"], mode="lines", line=dict(width=1), name="BB Low"), row=1, col=1)

    # COT Net Positions
    fig.add_trace(go.Bar(x=cot_df.index, y=cot_df["noncomm_net"], name="Non-Comm Net"), row=2, col=1)

    fig.update_layout(template="plotly_dark", height=800)
    return fig


# --- Opportunity Dashboard ---
def create_opportunity_dashboard(opportunities: dict):
    df = pd.DataFrame(opportunities).T
    df["health_gauge"] = df["health_gauge"].astype(float)
    df = df.sort_values(by="health_gauge", ascending=False)
    return df

# --- Streamlit App ---
def main():
    st.set_page_config(page_title="Market Dashboard", layout="wide")
    st.title("📊 Market Intelligence Dashboard")

    # Sidebar
    st.sidebar.header("Controls")
    selected_date = st.sidebar.date_input("Select Date", datetime.date.today())
    refresh_data = st.sidebar.button("🔄 Refresh Data")
    report_type = st.sidebar.selectbox("Report Type", ["Dashboard", "Detailed Analysis", "Newsletter"])

    # Newsletter generation option
    if st.sidebar.button("📰 Generate Newsletter"):
        st.session_state["generate_newsletter"] = True

    # Fetch & process data
    opportunities = {}
    for cot_name, ticker in ASSET_MAPPING.items():
        cot_df = fetch_cot_data(cot_name)
        price_df = fetch_yahooquery_data(ticker)
        if cot_df is None or price_df is None:
            continue

        merged_df = merge_cot_price(cot_df, price_df)
        if merged_df is None or merged_df.empty:
            continue

        health = calculate_health_gauge(cot_df, price_df)
        signals = generate_signals(merged_df, cot_df)
        trade_setup = generate_trade_setup(signals[0]["signal"], merged_df) if signals else {}

        opportunities[cot_name] = {
            "ticker": ticker,
            "health_gauge": health,
            "signals": signals,
            "trade_setup": trade_setup,
            "cot_df": cot_df,
            "price_df": merged_df,
        }

    # Tabs
    tabs = st.tabs(["📈 Market Dashboard", "🔍 Detailed Analysis", "📰 Market Newsletter"])

    # --- Dashboard Tab ---
    with tabs[0]:
        st.subheader("Market Opportunities")
        dashboard_df = create_opportunity_dashboard(opportunities)
        st.dataframe(dashboard_df[["health_gauge"]])

    # --- Detailed Analysis Tab ---
    with tabs[1]:
        asset_choice = st.selectbox("Select Asset", list(opportunities.keys()))
        asset_data = opportunities[asset_choice]
        fig = create_asset_chart(asset_data["price_df"], asset_data["cot_df"], asset_choice)
        st.plotly_chart(fig, use_container_width=True)

        st.write("**Signals:**", asset_data["signals"])
        st.write("**Trade Setup:**", asset_data["trade_setup"])
        st.write("**COT Latest:**", asset_data["cot_df"].tail(1).T)
        st.write("**Technical Indicators:**", asset_data["price_df"].tail(1).T)

    # --- Newsletter Tab ---
    with tabs[2]:
        if "generate_newsletter" in st.session_state and st.session_state["generate_newsletter"]:
            st.subheader("Generated Market Newsletter")

            all_newsletters = []
            for asset, data in opportunities.items():
                prompt = build_newsletter_prompt(asset, data["signals"], data["price_df"], data["cot_df"])
                newsletter = generate_market_newsletter(prompt)
                all_newsletters.append(f"### {asset}\n{newsletter}\n")

            st.markdown("\n\n".join(all_newsletters))
        else:
            st.info("Click '📰 Generate Newsletter' in the sidebar to create a report.")


if __name__ == "__main__":
    main()
