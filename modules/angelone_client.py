"""
AngelOne SmartAPI Client Wrapper
Handles authentication, session management, and all data fetching.
"""

import streamlit as st
import pyotp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import time
import logging

logger = logging.getLogger(__name__)

IST = pytz.timezone("Asia/Kolkata")

NIFTY_TOKEN = "99926000"
NIFTY_EXCHANGE = "NSE"

INTERVAL_MAP = {
    1: "ONE_MINUTE",
    2: "TWO_MINUTE",
    3: "THREE_MINUTE",
    5: "FIVE_MINUTE",
    10: "TEN_MINUTE",
    15: "FIFTEEN_MINUTE",
    30: "THIRTY_MINUTE",
    60: "ONE_HOUR",
}


def get_client():
    """
    Get or create an AngelOne SmartConnect client session.
    Uses st.session_state to cache the client across reruns.
    Returns the SmartConnect object or None on failure.
    """
    if "angel_client" in st.session_state and st.session_state.get("angel_client_valid", False):
        # Reuse existing authenticated session
        return st.session_state["angel_client"]

    try:
        from SmartApi import SmartConnect  # smartapi-python package

        secrets = st.secrets.get("angelone", {})
        api_key = secrets.get("api_key", "")
        client_id = secrets.get("client_id", "")
        password = secrets.get("password", "")
        totp_secret = secrets.get("totp_secret", "")

        if not all([api_key, client_id, password, totp_secret]):
            st.session_state["angel_client_valid"] = False
            return None

        obj = SmartConnect(api_key=api_key)
        totp = pyotp.TOTP(totp_secret).now()
        data = obj.generateSession(client_id, password, totp)

        if data and data.get("status"):
            st.session_state["angel_client"] = obj
            st.session_state["angel_client_valid"] = True
            st.session_state["angel_auth_token"] = data["data"]["jwtToken"]
            return obj
        else:
            st.session_state["angel_client_valid"] = False
            return None

    except ImportError:
        st.error("smartapi-python not installed. Run: pip install smartapi-python")
        return None
    except Exception as e:
        logger.error(f"AngelOne login error: {e}")
        st.session_state["angel_client_valid"] = False
        return None


def is_market_open() -> bool:
    """Check if NSE market is currently open (9:15 AM - 3:30 PM IST, Mon-Fri)."""
    now = datetime.now(IST)
    if now.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_open <= now <= market_close


def get_next_weekly_expiry() -> datetime:
    """
    Get the next weekly Nifty expiry date (every Thursday).
    If today is Thursday and market is still open, return today.
    """
    now = datetime.now(IST)
    today = now.date()
    # Thursday = weekday 3
    days_until_thursday = (3 - today.weekday()) % 7
    if days_until_thursday == 0:
        # Today is Thursday
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        if now > market_close:
            # Today's expiry has closed, get next Thursday
            days_until_thursday = 7
    expiry_date = today + timedelta(days=days_until_thursday)
    return datetime.combine(expiry_date, datetime.min.time()).replace(tzinfo=IST)


def get_expiry_countdown(expiry_dt: datetime) -> str:
    """Return human-readable countdown string to expiry."""
    now = datetime.now(IST)
    expiry_close = expiry_dt.replace(hour=15, minute=30, second=0)
    diff = expiry_close - now
    if diff.total_seconds() <= 0:
        return "Expired"
    total_seconds = int(diff.total_seconds())
    days = total_seconds // 86400
    hours = (total_seconds % 86400) // 3600
    minutes = (total_seconds % 3600) // 60

    if days == 0:
        return f"Expiry Today! {hours}h {minutes}m remaining"
    return f"{days}d {hours}h {minutes}m"


def get_expiry_string(expiry_dt: datetime) -> str:
    """Format expiry date as AngelOne API expects, e.g. '27JUN2024'."""
    return expiry_dt.strftime("%d%b%Y").upper()


@st.cache_data(ttl=10)
def fetch_candle_data(interval_minutes: int = 1, lookback_bars: int = 200) -> pd.DataFrame:
    """
    Fetch OHLCV candlestick data from AngelOne for NIFTY 50.
    Falls back to simulated data if API is unavailable.
    """
    try:
        obj = get_client()
        if obj is None:
            return _generate_mock_candles(interval_minutes, lookback_bars)

        interval_str = INTERVAL_MAP.get(interval_minutes, "ONE_MINUTE")
        now = datetime.now(IST)
        # Look back enough bars
        lookback_minutes = interval_minutes * lookback_bars
        from_dt = now - timedelta(minutes=lookback_minutes + 30)

        # If market hasn't opened today, go back to last trading day
        if now.hour < 9 or (now.hour == 9 and now.minute < 15):
            from_dt = (now - timedelta(days=1)).replace(hour=9, minute=15, second=0)

        from_str = from_dt.strftime("%Y-%m-%d %H:%M")
        to_str = now.strftime("%Y-%m-%d %H:%M")

        params = {
            "exchange": NIFTY_EXCHANGE,
            "symboltoken": NIFTY_TOKEN,
            "interval": interval_str,
            "fromdate": from_str,
            "todate": to_str,
        }
        response = obj.getCandleData(params)

        if response and response.get("status") and response.get("data"):
            raw = response["data"]
            df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
            df = df.sort_values("timestamp").reset_index(drop=True)
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df.dropna(inplace=True)
            return df
        else:
            return _generate_mock_candles(interval_minutes, lookback_bars)

    except Exception as e:
        logger.error(f"Candle data error: {e}")
        return _generate_mock_candles(interval_minutes, lookback_bars)


@st.cache_data(ttl=10)
def fetch_options_chain(expiry_str: str = None) -> pd.DataFrame:
    """
    Fetch options chain data including greeks from AngelOne.
    Returns DataFrame with strike, CE/PE OI, volume, IV, delta, gamma, theta, vega.
    Falls back to simulated data if API unavailable.
    """
    try:
        if expiry_str is None:
            expiry_dt = get_next_weekly_expiry()
            expiry_str = get_expiry_string(expiry_dt)

        obj = get_client()
        if obj is None:
            return _generate_mock_options_chain()

        response = obj.getOptionGreeks({"name": "NIFTY", "expirydate": expiry_str})

        if response and response.get("status") and response.get("data"):
            data = response["data"]
            rows = []
            for item in data:
                strike = float(item.get("strikePrice", 0))
                rows.append({
                    "strike": strike,
                    "ce_oi": int(item.get("CE", {}).get("openInterest", 0)),
                    "ce_volume": int(item.get("CE", {}).get("tradedVolume", 0)),
                    "ce_iv": float(item.get("CE", {}).get("impliedVolatility", 0)),
                    "ce_delta": float(item.get("CE", {}).get("delta", 0)),
                    "ce_gamma": float(item.get("CE", {}).get("gamma", 0)),
                    "ce_theta": float(item.get("CE", {}).get("theta", 0)),
                    "ce_vega": float(item.get("CE", {}).get("vega", 0)),
                    "ce_ltp": float(item.get("CE", {}).get("lastPrice", 0)),
                    "ce_bid": float(item.get("CE", {}).get("bidPrice", 0)),
                    "ce_ask": float(item.get("CE", {}).get("askPrice", 0)),
                    "pe_oi": int(item.get("PE", {}).get("openInterest", 0)),
                    "pe_volume": int(item.get("PE", {}).get("tradedVolume", 0)),
                    "pe_iv": float(item.get("PE", {}).get("impliedVolatility", 0)),
                    "pe_delta": float(item.get("PE", {}).get("delta", 0)),
                    "pe_gamma": float(item.get("PE", {}).get("gamma", 0)),
                    "pe_theta": float(item.get("PE", {}).get("theta", 0)),
                    "pe_vega": float(item.get("PE", {}).get("vega", 0)),
                    "pe_ltp": float(item.get("PE", {}).get("lastPrice", 0)),
                    "pe_bid": float(item.get("PE", {}).get("bidPrice", 0)),
                    "pe_ask": float(item.get("PE", {}).get("askPrice", 0)),
                })
            df = pd.DataFrame(rows)
            df.sort_values("strike", inplace=True)
            df.reset_index(drop=True, inplace=True)
            return df
        else:
            return _generate_mock_options_chain()

    except Exception as e:
        logger.error(f"Options chain error: {e}")
        return _generate_mock_options_chain()


def get_atm_strike(spot_price: float, step: int = 50) -> int:
    """Round spot price to nearest ATM strike (multiple of step)."""
    return int(round(spot_price / step) * step)


def get_strike_range(spot_price: float, n: int = 5, step: int = 50) -> list:
    """Get ATM ± n strikes."""
    atm = get_atm_strike(spot_price, step)
    return [atm + i * step for i in range(-n, n + 1)]


# ─── Mock / Simulation helpers ────────────────────────────────────────────────

def _generate_mock_candles(interval_minutes: int = 1, bars: int = 200) -> pd.DataFrame:
    """Generate realistic mock NIFTY candlestick data for demo/testing."""
    np.random.seed(int(time.time() / 60))  # Changes every minute
    now = datetime.now(IST).replace(tzinfo=None)

    # Start from today's 9:15 AM or lookback
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    if now < market_open:
        market_open = (now - timedelta(days=1)).replace(hour=9, minute=15, second=0, microsecond=0)

    base_price = 22000.0 + np.random.uniform(-500, 500)
    timestamps = []
    opens, highs, lows, closes, volumes = [], [], [], [], []

    price = base_price
    for i in range(bars):
        ts = market_open + timedelta(minutes=i * interval_minutes)
        # Skip non-market hours
        if ts.hour >= 15 and ts.minute > 30:
            ts = (ts + timedelta(days=1)).replace(hour=9, minute=15)
        if ts.weekday() >= 5:
            ts += timedelta(days=(7 - ts.weekday()))

        timestamps.append(ts)
        o = price
        change = np.random.normal(0, 0.3) * price / 100
        c = o + change
        h = max(o, c) + abs(np.random.normal(0, 0.15)) * price / 100
        l = min(o, c) - abs(np.random.normal(0, 0.15)) * price / 100
        v = int(np.random.randint(50000, 500000))
        opens.append(round(o, 2))
        highs.append(round(h, 2))
        lows.append(round(l, 2))
        closes.append(round(c, 2))
        volumes.append(v)
        price = c

    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })
    return df


def _generate_mock_options_chain() -> pd.DataFrame:
    """Generate mock options chain data around current NIFTY level."""
    np.random.seed(int(time.time() / 30))
    spot = 22000 + np.random.uniform(-300, 300)
    atm = get_atm_strike(spot)
    strikes = [atm + i * 50 for i in range(-10, 11)]

    rows = []
    for strike in strikes:
        moneyness = (spot - strike) / spot
        # Rough Black-Scholes approximations
        ce_iv = max(10, 15 - moneyness * 100 + abs(moneyness) * 50)
        pe_iv = max(10, 15 + moneyness * 100 + abs(moneyness) * 50)
        ce_delta = max(0.01, min(0.99, 0.5 + moneyness * 5))
        pe_delta = ce_delta - 1
        ce_gamma = max(0.0001, 0.005 - abs(moneyness) * 0.02)
        pe_gamma = ce_gamma
        ce_theta = -max(1, 20 - abs(strike - atm) / 10)
        pe_theta = -max(1, 20 - abs(strike - atm) / 10)
        ce_vega = max(0.1, 5 - abs(moneyness) * 20)
        pe_vega = ce_vega
        ce_ltp = max(1, (spot - strike) * ce_delta + np.random.uniform(5, 50))
        pe_ltp = max(1, (strike - spot) * abs(pe_delta) + np.random.uniform(5, 50))

        ce_oi = int(max(100, np.random.normal(500000, 200000)))
        pe_oi = int(max(100, np.random.normal(500000, 200000)))
        # More OI near ATM
        dist_factor = max(0.1, 1 - abs(strike - atm) / 500)
        ce_oi = int(ce_oi * dist_factor)
        pe_oi = int(pe_oi * dist_factor)

        rows.append({
            "strike": float(strike),
            "ce_oi": ce_oi,
            "ce_volume": int(ce_oi * 0.3 * np.random.uniform(0.5, 1.5)),
            "ce_iv": round(ce_iv, 2),
            "ce_delta": round(ce_delta, 4),
            "ce_gamma": round(ce_gamma, 6),
            "ce_theta": round(ce_theta, 2),
            "ce_vega": round(ce_vega, 4),
            "ce_ltp": round(ce_ltp, 2),
            "ce_bid": round(ce_ltp * 0.99, 2),
            "ce_ask": round(ce_ltp * 1.01, 2),
            "pe_oi": pe_oi,
            "pe_volume": int(pe_oi * 0.3 * np.random.uniform(0.5, 1.5)),
            "pe_iv": round(pe_iv, 2),
            "pe_delta": round(pe_delta, 4),
            "pe_gamma": round(pe_gamma, 6),
            "pe_theta": round(pe_theta, 2),
            "pe_vega": round(pe_vega, 4),
            "pe_ltp": round(pe_ltp, 2),
            "pe_bid": round(pe_ltp * 0.99, 2),
            "pe_ask": round(pe_ltp * 1.01, 2),
        })

    df = pd.DataFrame(rows)
    df.sort_values("strike", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


@st.cache_data(ttl=5)
def fetch_ltp(token: str = NIFTY_TOKEN) -> float:
    """Fetch the Last Traded Price for NIFTY 50."""
    try:
        obj = get_client()
        if obj is None:
            # Return mock price
            candles = fetch_candle_data(1, 2)
            return float(candles["close"].iloc[-1]) if not candles.empty else 22000.0

        response = obj.ltpData(NIFTY_EXCHANGE, "NIFTY 50", token)
        if response and response.get("status"):
            return float(response["data"]["ltp"])
        else:
            candles = fetch_candle_data(1, 2)
            return float(candles["close"].iloc[-1]) if not candles.empty else 22000.0
    except Exception as e:
        logger.error(f"LTP fetch error: {e}")
        candles = fetch_candle_data(1, 2)
        return float(candles["close"].iloc[-1]) if not candles.empty else 22000.0
