# fixed_cotrends_dashboard.py
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
from plotly.subplots import make_subplots
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
hf_client = None
try:
    hf_client = InferenceClient(provider="cerebras", api_key=os.getenv("HF_TOKEN"))
except Exception as e:
    logger.warning(f"HF client not initialized: {e}")

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

# Make alias for legacy code
ASSET_MAPPING = assets

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
                # Numeric conversions with coercion
                def to_num(col):
                    if col in df.columns:
                        return pd.to_numeric(df[col], errors="coerce")
                    else:
                        return pd.Series(np.nan, index=df.index)

                df["open_interest_all"] = to_num("open_interest_all")
                df["commercial_long"] = to_num("commercial_long_all")
                df["commercial_short"] = to_num("commercial_short_all")
                df["non_commercial_long"] = to_num("non_commercial_long_all")
                df["non_commercial_short"] = to_num("non_commercial_short_all")

                df["commercial_net"] = df["commercial_long"].fillna(0) - df["commercial_short"].fillna(0)
                df["non_commercial_net"] = df["non_commercial_long"].fillna(0) - df["non_commercial_short"].fillna(0)

                # Position % (safe denom)
                df["commercial_position_pct"] = np.where(
                    (df["commercial_long"] + df["commercial_short"]) > 0,
                    df["commercial_long"] / (df["commercial_long"] + df["commercial_short"]) * 100,
                    50.0,
                )
                df["non_commercial_position_pct"] = np.where(
                    (df["non_commercial_long"] + df["non_commercial_short"]) > 0,
                    df["non_commercial_long"] / (df["non_commercial_long"] + df["non_commercial_short"]) * 100,
                    50.0,
                )

                # Rolling z-scores (52-week ~ 52 reports)
                df = df.sort_values("report_date")
                df["commercial_net_zscore"] = (df["commercial_net"] - df["commercial_net"].rolling(52, min_periods=1).mean()) / df["commercial_net"].rolling(52, min_periods=1).std().replace(0, np.nan)
                df["non_commercial_net_zscore"] = (df["non_commercial_net"] - df["non_commercial_net"].rolling(52, min_periods=1).mean()) / df["non_commercial_net"].rolling(52, min_periods=1).std().replace(0, np.nan)

                return df.sort_values("report_date").reset_index(drop=True)
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
def fetch_yahooquery_data(ticker: str, start_date: str = None, end_date: str = None, max_attempts: int = 3) -> pd.DataFrame:
    """
    If start_date/end_date not provided, default to last 2 years.
    Returns DataFrame with a 'date' column (datetime) and typical OHLCV columns.
    """
    logger.info(f"Fetching Yahoo data for {ticker} from {start_date} to {end_date}")
    if end_date is None:
        end_date = datetime.date.today().isoformat()
    if start_date is None:
        start_date = (datetime.date.today() - datetime.timedelta(days=365 * 2)).isoformat()

    attempt = 0
    while attempt < max_attempts:
        try:
            t = Ticker(ticker)
            hist = t.history(start=start_date, end=end_date, interval="1d")
            if isinstance(hist, pd.DataFrame) and hist.empty:
                logger.warning(f"No price data for {ticker}")
                return pd.DataFrame()
            # yahooquery sometimes returns MultiIndex if multiple tickers specified
            if isinstance(hist.index, pd.MultiIndex):
                # If ticker present as level 0
                if ticker in hist.index.levels[0]:
                    hist = hist.loc[ticker]
                else:
                    hist = hist.reset_index(level=0, drop=True)

            hist = hist.reset_index()
            # Normalize column names to lowercase
            hist.columns = [c.lower() for c in hist.columns]
            if "date" not in hist.columns and "index" in hist.columns:
                hist = hist.rename(columns={"index": "date"})
            hist["date"] = pd.to_datetime(hist["date"])
            hist = hist.sort_values("date").reset_index(drop=True)

            # Calculate technical indicators
            hist = calculate_technical_indicators(hist)

            return hist
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

    close_col = "close" if "close" in df.columns else None
    high_col = "high" if "high" in df.columns else None
    low_col = "low" if "low" in df.columns else None

    if not all([close_col, high_col, low_col]):
        return df

    # RVOL
    vol_col = "volume" if "volume" in df.columns else None
    if vol_col is not None:
        df["rvol"] = df[vol_col] / df[vol_col].rolling(20, min_periods=1).mean()

    # RSI
    delta = df[close_col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14, min_periods=1).mean()
    avg_loss = loss.rolling(14, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # MAs
    df["sma20"] = df[close_col].rolling(20, min_periods=1).mean()
    df["sma50"] = df[close_col].rolling(50, min_periods=1).mean()
    df["sma200"] = df[close_col].rolling(200, min_periods=1).mean()

    # Bollinger Bands -> use bb_upper / bb_lower and store also bb_width
    df["bb_middle"] = df[close_col].rolling(20, min_periods=1).mean()
    df["bb_std"] = df[close_col].rolling(20, min_periods=1).std()
    df["bb_upper"] = df["bb_middle"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_middle"] - 2 * df["bb_std"]
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"].replace(0, np.nan)

    # ATR (named 'atr' for consistency)
    tr1 = df[high_col] - df[low_col]
    tr2 = (df[high_col] - df[close_col].shift()).abs()
    tr3 = (df[low_col] - df[close_col].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14, min_periods=1).mean()

    # Volatility annualized %
    df["volatility"] = df[close_col].pct_change().rolling(20, min_periods=1).std() * np.sqrt(252) * 100

    # 52-week high/low
    df["52w_high"] = df[close_col].rolling(252, min_periods=1).max()
    df["52w_low"] = df[close_col].rolling(252, min_periods=1).min()
    df["pct_from_52w_high"] = (df[close_col] / df["52w_high"] - 1) * 100
    df["pct_from_52w_low"] = (df[close_col] / df["52w_low"] - 1) * 100

    return df



# --- Merge COT and Price Data ---
def merge_cot_price(cot_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge COT and price data using 'date' column.
    Fixes MergeError caused by timezone-aware vs naive datetime.
    """
    if cot_df.empty or price_df.empty:
        return pd.DataFrame()

    # Ensure datetime naive for merge
    cot_df = cot_df.copy()
    price_df = price_df.copy()

    if pd.api.types.is_datetime64tz_dtype(cot_df["report_date"]):
        cot_df["report_date"] = cot_df["report_date"].dt.tz_localize(None)
    cot_df = cot_df.rename(columns={"report_date": "date"})
    price_df["date"] = pd.to_datetime(price_df["date"]).dt.tz_localize(None)

    # Ensure sorted for merge_asof
    cot_df = cot_df.sort_values("date")
    price_df = price_df.sort_values("date")

    # Forward fill missing COT data for price dates
    merged = pd.merge_asof(price_df, cot_df, on="date", direction="backward")
    return merged.reset_index(drop=True)

# --- Health Gauge Calculation ---
def calculate_health_gauge(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine RVol, RSI, ATR, and Bollinger Bands width to compute a health gauge score.
    """
    if df.empty:
        return df

    # Normalize features 0-1
    df = df.copy()
    df["rvol_norm"] = df["rvol"] / df["rvol"].rolling(20, min_periods=1).max()
    df["rsi_norm"] = df["rsi"] / 100
    df["atr_norm"] = df["atr"] / df["atr"].rolling(20, min_periods=1).max()
    df["bb_width_norm"] = df["bb_width"] / df["bb_width"].rolling(20, min_periods=1).max()

    # Health gauge: lower volatility + higher liquidity = higher score
    df["health_gauge"] = (1 - df["rvol_norm"]) * 0.4 + (1 - df["atr_norm"]) * 0.3 + (1 - df["bb_width_norm"]) * 0.3
    df["health_gauge"] = df["health_gauge"].clip(0, 1)

    return df

# --- Signal Generation ---
def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate simple buy/sell signals based on RSI and COT net positions.
    """
    if df.empty:
        return df

    df = df.copy()
    df["signal"] = "HOLD"

    # RSI oversold/overbought + COT extremes
    df.loc[(df["rsi"] < 30) & (df["non_commercial_net_zscore"] < -1), "signal"] = "BUY"
    df.loc[(df["rsi"] > 70) & (df["non_commercial_net_zscore"] > 1), "signal"] = "SELL"
    return df

# --- Streamlit UI ---
def main():
    st.set_page_config(layout="wide")
    st.title("📊 Market Intelligence Dashboard")
    st.markdown("COT, Price, and Technical Indicator Analysis")

    # Controls
    asset = st.selectbox("Select Asset", list(ASSET_MAPPING.keys()))
    start_date = st.date_input("Start Date", datetime.date.today() - datetime.timedelta(days=365))
    end_date = st.date_input("End Date", datetime.date.today())

    # Fetch data
    cot_df = fetch_cot_data(asset)
    price_df = fetch_yahooquery_data(ASSET_MAPPING[asset], start_date=start_date.isoformat(), end_date=end_date.isoformat())

    if cot_df.empty or price_df.empty:
        st.warning("Data not available for selected asset/date range.")
        return

    # Merge
    merged_df = merge_cot_price(cot_df, price_df)
    merged_df = calculate_health_gauge(merged_df)
    merged_df = generate_signals(merged_df)

    # Plot health gauge
    st.subheader("Health Gauge & RSI")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=merged_df["date"], y=merged_df["health_gauge"], name="Health Gauge", line=dict(color="green")), secondary_y=False)
    fig.add_trace(go.Scatter(x=merged_df["date"], y=merged_df["rsi"], name="RSI", line=dict(color="blue")), secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

    # Show signals
    st.subheader("Signals")
    st.dataframe(merged_df[["date", "close", "signal", "health_gauge", "rsi"]].tail(50))

    # Optionally: show full merged dataframe
    if st.checkbox("Show full merged dataframe"):
        st.dataframe(merged_df)

if __name__ == "__main__":
    main()



