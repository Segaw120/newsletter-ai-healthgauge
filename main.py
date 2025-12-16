import streamlit as st
import pandas as pd
import numpy as np
import datetime
import time
import os
import logging
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sodapy import Socrata
from yahooquery import Ticker
from scipy import stats

# NEW: OpenAI-compatible client to route to Hugging Face router
from openai import OpenAI

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Constants and Setup ---
SODAPY_APP_TOKEN = "PP3ezxaUTiGforvvbBUGzwRx7"
client = Socrata("publicreporting.cftc.gov", SODAPY_APP_TOKEN, timeout=60)

# ---- Insert / replace your HF client init with this block ----
HF_MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
HF_TOKEN = os.environ.get("HF_TOKEN")  # ensure this is exported in your env

# tiny wrapper so your existing generate_market_newsletter() doesn't change
class _HFClientWrapper:
    def __init__(self, openai_client, model_id):
        self._client = openai_client
        self._model = model_id

    def chat_completion(self, messages, max_tokens=500, temperature=0.7, **kwargs):
        """
        Mirrors the minimal call shape your function expects.
        Forwards call to OpenAI(client) -> HF router.
        """
        return self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )

# Initialize OpenAI client pointed at Hugging Face router
try:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN environment variable not set")

    _openai_hf = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=HF_TOKEN,
    )
    hf_client = _HFClientWrapper(_openai_hf, HF_MODEL_ID)
    logging.getLogger(__name__).info(f"hf_client initialized and locked to model: {HF_MODEL_ID}")

except Exception as e:
    hf_client = None
    logging.getLogger(__name__).warning(f"hf_client not initialized: {e}")
# ---------------------------------------------------------------

# --- Asset mapping (COT market names -> Yahoo futures tickers) ---
ASSET_MAPPING = {
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
    "E-MINI S&P FINANCIAL INDEX - CHICAGO MERCANTILE EXCHANGE": "ES=F",
    "NATURAL GAS - NEW YORK MERCANTILE EXCHANGE": "NG=F",
    "CORN - CHICAGO BOARD OF TRADE": "ZC=F",
    "SOYBEANS - CHICAGO BOARD OF TRADE": "ZS=F",
}

# --- Fetch COT Data ---
@st.cache_data(ttl=3600)
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
                    
                    # Calculate position percentages
                    df["commercial_position_pct"] = (df["commercial_long"] / 
                                                    (df["commercial_long"] + df["commercial_short"])) * 100
                    df["non_commercial_position_pct"] = (df["non_commercial_long"] / 
                                                        (df["non_commercial_long"] + df["non_commercial_short"])) * 100
                    
                    # Calculate z-scores
                    df["commercial_net_zscore"] = (df["commercial_net"] - 
                                                 df["commercial_net"].rolling(52).mean()) / df["commercial_net"].rolling(52).std()
                    df["non_commercial_net_zscore"] = (df["non_commercial_net"] - 
                                                     df["non_commercial_net"].rolling(52).mean()) / df["non_commercial_net"].rolling(52).std()
                    
                except KeyError as e:
                    logger.warning(f"Missing COT columns for {market_name}: {e}")
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

# --- Fetch Price Data ---
@st.cache_data(ttl=3600)
def fetch_yahooquery_data(ticker: str, max_attempts: int = 3) -> pd.DataFrame:
    logger.info(f"Fetching Yahoo data for {ticker}")
    start_date = (datetime.datetime.now() - datetime.timedelta(days=730)).strftime("%Y-%m-%d")
    end_date = datetime.datetime.now().strftime("%Y-%m-%d")
    
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
    else:
        df["rvol"] = np.nan
    
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
    
    # Forward fill COT data
    for col in cot_columns[1:]:
        merged[col] = merged[col].ffill()
    
    return merged

# --- Calculate Health Gauge ---
def calculate_health_gauge(merged_df: pd.DataFrame) -> float:
    if merged_df.empty:
        return np.nan
    
    latest = merged_df.tail(1).iloc[0]
    recent = merged_df.tail(90).copy()
    
    close_col = "close" if "close" in recent.columns else ("Close" if "Close" in recent.columns else None)
    
    if close_col is None:
        return np.nan
    
    scores = []
    
    # 1. Commercial net position score (25%)
    if "commercial_net_zscore" in latest and not pd.isna(latest["commercial_net_zscore"]):
        comm_score = max(0, min(1, 0.5 - latest["commercial_net_zscore"]/4))
        scores.append((comm_score, 0.25))
    
    # 2. Trend alignment score (20%)
    if all(x in recent.columns for x in [close_col, "sma20", "sma50", "sma200"]):
        last_close = latest[close_col]
        trend_signals = [
            last_close > latest["sma20"],
            latest["sma20"] > latest["sma50"],
            latest["sma50"] > latest["sma200"],
        ]
        trend_score = sum(trend_signals) / len(trend_signals)
        scores.append((trend_score, 0.20))
    
    # 3. Momentum score (15%)
    if "rsi" in recent.columns and not pd.isna(latest["rsi"]):
        rsi = latest["rsi"]
        if rsi < 30:
            rsi_score = 0.3
        elif rsi > 70:
            rsi_score = 0.7
        else:
            rsi_score = 0.5 + (rsi - 50) / 100
        scores.append((rsi_score, 0.15))
    
    # 4. Volatility and volume score (15%)
    vol_score = 0.5
    if "bb_width" in recent.columns and "rvol" in recent.columns:
        bb_width_percentile = stats.percentileofscore(
            recent["bb_width"].dropna(), latest["bb_width"]) / 100 if not pd.isna(latest["bb_width"]) else 0.5
        bb_score = 1 - bb_width_percentile
        rvol_score = min(1.0, latest["rvol"] / 2.0) if not pd.isna(latest["rvol"]) else 0.5
        vol_score = 0.7 * bb_score + 0.3 * rvol_score
        scores.append((vol_score, 0.15))
    
    # 5. Distance from 52-week high/low score (15%)
    if "pct_from_52w_high" in recent.columns and "pct_from_52w_low" in recent.columns:
        high_score = 1 - (abs(latest["pct_from_52w_high"]) / 100)
        high_score = max(0, min(1, high_score))
        low_score = min(1, latest["pct_from_52w_low"] / 100)
        low_score = max(0, min(1, low_score))
        dist_score = 0.7 * high_score + 0.3 * low_score
        scores.append((dist_score, 0.15))
    
    # 6. Open interest score (10%)
    if "open_interest_all" in recent.columns:
        oi = recent["open_interest_all"].dropna()
        if not oi.empty and not pd.isna(latest["open_interest_all"]):
            oi_pctile = stats.percentileofscore(oi, latest["open_interest_all"]) / 100
            scores.append((oi_pctile, 0.10))
    
    if not scores:
        return 5.0
    
    weighted_sum = sum(score * weight for score, weight in scores)
    total_weight = sum(weight for _, weight in scores)
    
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
    
    # COT commercial position signals
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
    
    # Price trend signals
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
    
    # RSI signals
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
    
    # Bollinger Band signals
    if all(x in latest for x in ["bb_upper", "bb_lower", close_col]):
        if latest[close_col] > latest["bb_upper"]:
            bearish_points += 1
            signal_reasons.append("Price above upper Bollinger Band")
        elif latest[close_col] < latest["bb_lower"]:
            bullish_points += 1
            signal_reasons.append("Price below lower Bollinger Band")
    
    # Volume signals
    if "rvol" in latest and not pd.isna(latest["rvol"]):
        if latest["rvol"] > 1.5 and len(recent) > 1:
            price_change = latest[close_col] - recent.iloc[-2][close_col]
            if price_change > 0:
                bullish_points += 1
                signal_reasons.append("High volume on price advance")
            elif price_change < 0:
                bearish_points += 1
                signal_reasons.append("High volume on price decline")
    
    # 52-week signals
    if "pct_from_52w_high" in latest and "pct_from_52w_low" in latest:
        if latest["pct_from_52w_high"] > -5:
            bullish_points += 1
            signal_reasons.append("Price near 52-week high")
        elif latest["pct_from_52w_low"] < 10:
            bearish_points += 1
            signal_reasons.append("Price near 52-week low")
    
    # Determine signal
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
    
    return {
        "signal": signal,
        "strength": strength,
        "reasoning": "; ".join(signal_reasons) if signal_reasons else "No strong signals detected"
    }

# --- AI Newsletter Generation ---
def build_newsletter_prompt(asset: str, signal_data: dict, merged_df: pd.DataFrame) -> str:
    if merged_df.empty:
        return f"No data available for {asset}."
    
    latest = merged_df.iloc[-1]
    close_col = "close" if "close" in latest else "Close"
    
    prompt = f\"\"\"Write a concise professional market analysis for {asset.split(' - ')[0]}.

Current Data:
- Price: ${latest.get(close_col, 0):.2f}
- RSI: {latest.get('rsi', 0):.1f}
- Signal: {signal_data['signal']} (Strength: {signal_data['strength']}/5)
- Key Factors: {signal_data['reasoning']}
- Commercial Positioning Z-Score: {latest.get('commercial_net_zscore', 0):.2f}
- 52-Week Range: {latest.get('pct_from_52w_low', 0):.1f}% from low, {latest.get('pct_from_52w_high', 0):.1f}% from high

Write a 3-paragraph analysis covering:
1. Current market position and trend
2. Key technical and COT indicators
3. Trading outlook and risk factors

Keep it professional and actionable. No fluff.\"\"\"
    
    return prompt.strip()

def generate_market_newsletter(prompt: str) -> str:
    if hf_client is None:
        return "AI newsletter generation unavailable (API key not configured)"
    
    try:
        response = hf_client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Newsletter generation failed: {e}")
        return f"Error generating newsletter: {str(e)}"

# --- Visualization ---
def create_asset_chart(merged_df: pd.DataFrame, asset_name: str):
    if merged_df.empty:
        return None
    
    close_col = "close" if "close" in merged_df.columns else "Close"
    high_col = "high" if "high" in merged_df.columns else "High"
    low_col = "low" if "low" in merged_df.columns else "Low"
    open_col = "open" if "open" in merged_df.columns else "Open"
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.5, 0.25, 0.25],
        vertical_spacing=0.03,
        subplot_titles=(
            f"{asset_name.split(' - ')[0]} Price Analysis",
            "RSI",
            "COT Commercial Net Position Z-Score"
        )
    )
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=merged_df["date"],
        open=merged_df[open_col],
        high=merged_df[high_col],
        low=merged_df[low_col],
        close=merged_df[close_col],
        name="Price"
    ), row=1, col=1)
    
    # Moving averages
    fig.add_trace(go.Scatter(
        x=merged_df["date"], y=merged_df["sma50"],
        mode="lines", name="SMA50", line=dict(color="orange", width=1)
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=merged_df["date"], y=merged_df["sma200"],
        mode="lines", name="SMA200", line=dict(color="red", width=1)
    ), row=1, col=1)
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(
        x=merged_df["date"], y=merged_df["bb_upper"],
        mode="lines", name="BB Upper", line=dict(color="gray", width=1, dash="dash"),
        showlegend=False
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=merged_df["date"], y=merged_df["bb_lower"],
        mode="lines", name="BB Lower", line=dict(color="gray", width=1, dash="dash"),
        fill='tonexty', fillcolor='rgba(128,128,128,0.1)',
        showlegend=False
    ), row=1, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(
        x=merged_df["date"], y=merged_df["rsi"],
        mode="lines", name="RSI", line=dict(color="purple")
    ), row=2, col=1)
    
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1, opacity=0.5)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1, opacity=0.5)
    
    # COT Z-Score
    fig.add_trace(go.Scatter(
        x=merged_df["date"], y=merged_df["commercial_net_zscore"],
        mode="lines", name="Commercial Z-Score", line=dict(color="blue")
    ), row=3, col=1)
    
    fig.add_hline(y=0, line_dash="solid", line_color="white", row=3, col=1, opacity=0.3)
    fig.add_hline(y=1.5, line_dash="dash", line_color="red", row=3, col=1, opacity=0.5)
    fig.add_hline(y=-1.5, line_dash="dash", line_color="green", row=3, col=1, opacity=0.5)
    
    fig.update_layout(
        template="plotly_dark",
        height=900,
        showlegend=True,
        hovermode="x unified"
    )
    
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="Z-Score", row=3, col=1)
    
    return fig

# --- Streamlit App ---
def main():
    st.set_page_config(page_title="Market Intelligence Dashboard", layout="wide", page_icon="📊")
    
    # Custom CSS
    st.markdown("""
        <style>
        .big-font {
            font-size:20px !important;
            font-weight: bold;
        }
        .metric-card {
            background-color: #1e1e1e;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #333;
        }
        .newsletter-section {
            background-color: #2d2d2d;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
            border-left: 4px solid #4CAF50;
        }
        </style>
        """, unsafe_allow_html=True)
    
    st.title("📊 Market Intelligence Dashboard")
    st.markdown("*Real-time COT analysis with AI-powered insights*")
    
    # Sidebar
    st.sidebar.header("⚙️ Controls")
    
    # Select assets to analyze
    selected_assets = st.sidebar.multiselect(
        "Select Assets to Analyze",
        list(ASSET_MAPPING.keys()),
        default=list(ASSET_MAPPING.keys())[:5]
    )
    
    if not selected_assets:
        st.warning("Please select at least one asset to analyze")
        return
    
    refresh_data = st.sidebar.button("🔄 Refresh All Data", type="primary")
    
    # Clear cache if refresh requested
    if refresh_data:
        st.cache_data.clear()
        st.rerun()
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📈 Dashboard", "🔍 Detailed Analysis", "📰 AI Newsletter"])
    
    # Fetch data with progress bar
    with st.spinner("Fetching market data..."):
        progress_bar = st.progress(0)
        opportunities = {}
        
        for idx, cot_name in enumerate(selected_assets):
            ticker = ASSET_MAPPING[cot_name]
            
            try:
                # Fetch data
                cot_df = fetch_cot_data(cot_name)
                price_df = fetch_yahooquery_data(ticker)
                
                if cot_df.empty or price_df.empty:
                    logger.warning(f"Skipping {cot_name} - insufficient data")
                    continue
                
                # Merge and analyze
                merged_df = merge_cot_price(cot_df, price_df)
                if merged_df.empty:
                    continue
                
                health = calculate_health_gauge(merged_df)
                signals = generate_signals(merged_df)
                
                opportunities[cot_name] = {
                    "ticker": ticker,
                    "health_gauge": health,
                    "signal": signals["signal"],
                    "signal_strength": signals["strength"],
                    "signal_reasoning": signals["reasoning"],
                    "merged_df": merged_df,
                    "latest_price": merged_df.iloc[-1].get("close", merged_df.iloc[-1].get("Close", 0)),
                }
                
            except Exception as e:
                logger.error(f"Error processing {cot_name}: {e}")
                continue
            
            progress_bar.progress((idx + 1) / len(selected_assets))
        
        progress_bar.empty()
    
    if not opportunities:
        st.error("No data available for selected assets. Please try different selections.")
        return
    
    # --- Dashboard Tab ---
    with tab1:
        st.subheader("Market Opportunity Dashboard")
        
        # Create dashboard dataframe
        dashboard_data = []
        for asset, data in opportunities.items():
            dashboard_data.append({
                "Asset": asset.split(" - ")[0],
                "Price": f"${data['latest_price']:.2f}",
                "Health Score": f"{data['health_gauge']:.1f}/10",
                "Signal": data['signal'],
                "Strength": "⭐" * data['signal_strength'],
            })
        
        df_display = pd.DataFrame(dashboard_data)
        
        # Color code signals
        def color_signal(val):
            if "BUY" in val:
                return 'background-color: #1e4620; color: white'
            elif "SELL" in val:
                return 'background-color: #4a1a1a; color: white'
            else:
                return 'background-color: #3a3a3a; color: white'
        
        styled_df = df_display.style.applymap(color_signal, subset=['Signal'])
        st.dataframe(styled_df, use_container_width=True, height=400)
        
        # Top opportunities
        st.subheader("🎯 Top Opportunities")
        sorted_opps = sorted(opportunities.items(), key=lambda x: x[1]['health_gauge'], reverse=True)
        
        cols = st.columns(min(3, len(sorted_opps)))
        for idx, (asset, data) in enumerate(sorted_opps[:3]):
            with cols[idx]:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric(
                    label=asset.split(" - ")[0],
                    value=f"${data['latest_price']:.2f}",
                    delta=f"Health: {data['health_gauge']:.1f}/10"
                )
                st.markdown(f"**{data['signal']}** {'⭐' * data['signal_strength']}")
                st.markdown('</div>', unsafe_allow_html=True)
    
    # --- Detailed Analysis Tab ---
    with tab2:
        st.subheader("Detailed Technical Analysis")
        
        selected_asset = st.selectbox(
            "Select Asset for Detailed View",
            list(opportunities.keys()),
            format_func=lambda x: x.split(" - ")[0]
        )
        
        if selected_asset:
            data = opportunities[selected_asset]
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Price", f"${data['latest_price']:.2f}")
            with col2:
                st.metric("Health Score", f"{data['health_gauge']:.1f}/10")
            with col3:
                st.metric("Signal", data['signal'])
            with col4:
                st.metric("Strength", "⭐" * data['signal_strength'])
            
            st.markdown(f"**Analysis:** {data['signal_reasoning']}")
            
            # Chart
            fig = create_asset_chart(data['merged_df'], selected_asset)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Latest technical indicators
            st.subheader("Technical Indicators")
            latest = data['merged_df'].iloc[-1]
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Trend Indicators**")
                st.write(f"• RSI: {latest.get('rsi', 0):.1f}")
                st.write(f"• SMA 20: ${latest.get('sma20', 0):.2f}")
                st.write(f"• SMA 50: ${latest.get('sma50', 0):.2f}")
                st.write(f"• SMA 200: ${latest.get('sma200', 0):.2f}")
            
            with col2:
                st.markdown("**COT & Volatility**")
                st.write(f"• Commercial Z-Score: {latest.get('commercial_net_zscore', 0):.2f}")
                st.write(f"• Volatility: {latest.get('volatility', 0):.1f}%")
                st.write(f"• ATR: ${latest.get('atr14', 0):.2f}")
                st.write(f"• From 52W High: {latest.get('pct_from_52w_high', 0):.1f}%")
    
    # --- Newsletter Tab ---
    with tab3:
        st.subheader("📰 AI-Generated Market Newsletter")
        st.markdown(f"*Generated on {datetime.datetime.now().strftime('%B %d, %Y at %I:%M %p')}*")
        
        if st.button("🤖 Generate Newsletter", type="primary"):
            with st.spinner("Generating AI analysis..."):
                for asset, data in opportunities.items():
                    st.markdown('<div class="newsletter-section">', unsafe_allow_html=True)
                    st.markdown(f"### 📊 {asset.split(' - ')[0]}")
                    st.markdown(f"**Current Price:** ${data['latest_price']:.2f} | **Signal:** {data['signal']} {'⭐' * data['signal_strength']} | **Health:** {data['health_gauge']:.1f}/10")
                    st.markdown("---")
                    
                    prompt = build_newsletter_prompt(asset, 
                        {"signal": data['signal'], "strength": data['signal_strength'], "reasoning": data['signal_reasoning']},
                        data['merged_df'])
                    
                    newsletter = generate_market_newsletter(prompt)
                    st.markdown(newsletter)
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown("")
        else:
            st.info("Click 'Generate Newsletter' to create AI-powered market analysis for all selected assets")

if __name__ == "__main__":
    main()
