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
        if not st.session_state.get("_smartapi_warn_shown"):
            st.session_state["_smartapi_warn_shown"] = True
            st.warning("smartapi-python not installed — running in DEMO mode. Run: pip install smartapi-python")
        st.session_state["angel_client_valid"] = False
        return None
    except Exception as e:
        logger.error(f"AngelOne login error: {e}")
        st.session_state["angel_client_valid"] = False
        return None


def is_connected() -> bool:
    """Return True if a live authenticated AngelOne API session is active."""
    # Trigger a connection attempt if not yet tried this run
    if "angel_client_valid" not in st.session_state:
        get_client()
    return bool(st.session_state.get("angel_client_valid", False))


def get_data_source() -> str:
    """Return 'LIVE' if connected to AngelOne, otherwise 'DEMO'."""
    return "LIVE" if is_connected() else "DEMO"


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
    Get the next weekly Nifty expiry date dynamically from AngelOne API.
    Searches NFO scrips for NIFTY options and returns the nearest upcoming expiry.
    Falls back to session_state cached value, then to nearest weekday if API unavailable.
    """
    now = datetime.now(IST)
    today = now.date()

    # Try AngelOne API first
    try:
        obj = get_client()
        if obj is not None:
            response = obj.searchScrip("NFO", "NIFTY")
            if response and response.get("status") and response.get("data"):
                expiry_dates = set()
                for scrip in response["data"]:
                    # Only weekly options (not monthly futures/options with far dates)
                    name = scrip.get("tradingsymbol", "")
                    expiry_str = scrip.get("expiry", "")
                    if not expiry_str or "NIFTY" not in name:
                        continue
                    try:
                        exp_date = datetime.strptime(expiry_str, "%d%b%Y").date()
                        if exp_date >= today:
                            expiry_dates.add(exp_date)
                    except Exception:
                        try:
                            exp_date = datetime.strptime(expiry_str, "%Y-%m-%d").date()
                            if exp_date >= today:
                                expiry_dates.add(exp_date)
                        except Exception:
                            pass

                if expiry_dates:
                    # Pick the nearest upcoming expiry
                    nearest = min(expiry_dates)
                    # If nearest is today and market closed, pick the next one
                    if nearest == today:
                        mkt_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
                        if now > mkt_close and len(expiry_dates) > 1:
                            nearest = sorted(expiry_dates)[1]
                    expiry_dt = datetime.combine(nearest, datetime.min.time()).replace(tzinfo=IST)
                    # Cache for fallback use
                    st.session_state["_last_known_expiry"] = expiry_dt
                    return expiry_dt
    except Exception as e:
        logger.debug(f"Expiry fetch from API failed: {e}")

    # Fallback 1: use last known expiry from session_state if still valid
    cached = st.session_state.get("_last_known_expiry")
    if cached is not None:
        cached_date = cached.date() if hasattr(cached, "date") else cached
        if cached_date >= today:
            return cached

    # Fallback 2: scan next 7 days — pick nearest weekday that is Mon–Fri
    # (expiry is always a weekday; we don't know which day without API,
    # so return the soonest non-weekend day within the next 7 days)
    for delta in range(0, 8):
        candidate = today + timedelta(days=delta)
        if candidate.weekday() < 5:  # Mon=0 … Fri=4
            if candidate == today:
                mkt_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
                if now <= mkt_close:
                    return datetime.combine(candidate, datetime.min.time()).replace(tzinfo=IST)
            else:
                return datetime.combine(candidate, datetime.min.time()).replace(tzinfo=IST)

    # Absolute fallback
    return datetime.combine(today + timedelta(days=1), datetime.min.time()).replace(tzinfo=IST)


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


_EMPTY_CANDLES = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])


@st.cache_data(ttl=10)
def fetch_candle_data(interval_minutes: int = 1, lookback_bars: int = 200) -> pd.DataFrame:
    """
    Fetch real OHLCV candlestick data from AngelOne for NIFTY 50.
    When the market is closed, this returns the LAST trading day's session
    (the API request window is widened to bridge weekends/holidays).
    Returns an empty DataFrame if not connected — no simulated data.
    """
    obj = get_client()
    if obj is None:
        return _EMPTY_CANDLES.copy()

    try:
        interval_str = INTERVAL_MAP.get(interval_minutes, "ONE_MINUTE")
        now = datetime.now(IST)

        # Widen the window enough to always include the last completed session,
        # even across a weekend or a string of holidays (look back up to 6 days),
        # while still requesting enough history for the chosen interval.
        lookback_minutes = interval_minutes * lookback_bars
        from_dt = now - timedelta(minutes=lookback_minutes + 30)
        earliest = now - timedelta(days=6)
        if from_dt > earliest:
            from_dt = earliest

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

        return _EMPTY_CANDLES.copy()

    except Exception as e:
        logger.error(f"Candle data error: {e}")
        return _EMPTY_CANDLES.copy()


@st.cache_data(ttl=10)
def fetch_options_chain(expiry_str: str = None) -> pd.DataFrame:
    """
    Fetch options chain data including greeks from AngelOne.
    Returns DataFrame with strike, CE/PE OI, volume, IV, delta, gamma, theta, vega.
    Returns an empty DataFrame if not connected — no simulated data.
    """
    try:
        if expiry_str is None:
            expiry_dt = get_next_weekly_expiry()
            expiry_str = get_expiry_string(expiry_dt)

        obj = get_client()
        if obj is None:
            return pd.DataFrame()

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

        return pd.DataFrame()

    except Exception as e:
        logger.error(f"Options chain error: {e}")
        return pd.DataFrame()


def get_atm_strike(spot_price: float, step: int = 50) -> int:
    """Round spot price to nearest ATM strike (multiple of step)."""
    return int(round(spot_price / step) * step)


def get_strike_range(spot_price: float, n: int = 5, step: int = 50) -> list:
    """Get ATM ± n strikes."""
    atm = get_atm_strike(spot_price, step)
    return [atm + i * step for i in range(-n, n + 1)]


def _last_trading_day(ref: datetime) -> datetime:
    """Return the most recent trading day (Mon–Fri) on or before ref's date."""
    d = ref
    # If before market open today, the latest completed session is the prior day
    if d.weekday() >= 5:  # weekend -> roll back to Friday
        d = d - timedelta(days=(d.weekday() - 4))
    elif d.hour < 9 or (d.hour == 9 and d.minute < 15):
        d = d - timedelta(days=1)
        while d.weekday() >= 5:
            d = d - timedelta(days=1)
    return d


@st.cache_data(ttl=5)
def fetch_ltp(token: str = NIFTY_TOKEN) -> float:
    """
    Fetch the Last Traded Price for NIFTY 50 from AngelOne.
    When the market is closed this returns the last traded price (last close).
    Returns 0.0 if not connected — no simulated price.
    """
    obj = get_client()
    if obj is None:
        return 0.0

    try:
        response = obj.ltpData(NIFTY_EXCHANGE, "NIFTY 50", token)
        if response and response.get("status"):
            return float(response["data"]["ltp"])
        # Fall back to the last real candle close
        candles = fetch_candle_data(1, 2)
        return float(candles["close"].iloc[-1]) if not candles.empty else 0.0
    except Exception as e:
        logger.error(f"LTP fetch error: {e}")
        candles = fetch_candle_data(1, 2)
        return float(candles["close"].iloc[-1]) if not candles.empty else 0.0
