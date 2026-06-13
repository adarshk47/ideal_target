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


def _read_secrets() -> dict:
    """
    Read AngelOne credentials from st.secrets, supporting either section name
    ([angel_one] or [angelone]) and either 'mpin' or 'password' for login.
    """
    section = {}
    for key in ("angel_one", "angelone", "ANGEL_ONE", "ANGELONE"):
        try:
            if key in st.secrets:
                section = st.secrets[key]
                break
        except Exception:
            continue
    # Fall back to a flat layout (keys at top level)
    if not section:
        section = st.secrets

    def g(*names):
        for n in names:
            try:
                if n in section and section[n]:
                    return str(section[n])
            except Exception:
                pass
        return ""

    return {
        "api_key": g("api_key", "apikey", "key"),
        "client_id": g("client_id", "clientid", "client_code", "clientcode"),
        # AngelOne login now uses MPIN; fall back to password for older setups
        "login_pwd": g("mpin", "pin", "password"),
        "totp_secret": g("totp_secret", "totp", "totp_key"),
    }


def get_client(force: bool = False):
    """
    Get or create an AngelOne SmartConnect client session.
    Caches the client in st.session_state. Stores the last error message in
    st.session_state['angel_error'] so the UI can show why login failed.
    Pass force=True to retry a fresh login (used by the Connect button).
    Returns the SmartConnect object or None on failure.
    """
    if force:
        st.session_state.pop("angel_client", None)
        st.session_state["angel_client_valid"] = False

    if st.session_state.get("angel_client") is not None and \
            st.session_state.get("angel_client_valid", False):
        return st.session_state["angel_client"]

    st.session_state["angel_error"] = ""
    try:
        from SmartApi import SmartConnect  # smartapi-python package
    except ImportError as e:
        st.session_state["angel_client_valid"] = False
        st.session_state["angel_error"] = (
            f"smartapi-python (or a dependency like logzero) not installed: {e}"
        )
        return None

    creds = _read_secrets()
    missing = [k for k in ("api_key", "client_id", "login_pwd", "totp_secret")
               if not creds.get(k)]
    if missing:
        st.session_state["angel_client_valid"] = False
        st.session_state["angel_error"] = (
            "Missing credentials in secrets: " + ", ".join(missing) +
            ". Expected a [angel_one] section with api_key, client_id, "
            "mpin (or password) and totp_secret."
        )
        return None

    try:
        obj = SmartConnect(api_key=creds["api_key"])
        totp = pyotp.TOTP(creds["totp_secret"]).now()
        data = obj.generateSession(creds["client_id"], creds["login_pwd"], totp)

        if data and data.get("status"):
            try:
                obj.getfeedToken()
            except Exception:
                pass
            st.session_state["angel_client"] = obj
            st.session_state["angel_client_valid"] = True
            st.session_state["angel_auth_token"] = data["data"]["jwtToken"]
            st.session_state["angel_error"] = ""
            return obj

        # Login returned a failure payload — surface the API message
        msg = ""
        if isinstance(data, dict):
            msg = data.get("message") or data.get("errorcode") or str(data)
        st.session_state["angel_client_valid"] = False
        st.session_state["angel_error"] = f"Login failed: {msg}"
        return None

    except Exception as e:
        logger.error(f"AngelOne login error: {e}")
        st.session_state["angel_client_valid"] = False
        st.session_state["angel_error"] = f"Login error: {e}"
        return None


def get_last_error() -> str:
    """Return the last connection error message (empty string if none)."""
    return st.session_state.get("angel_error", "")


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


def _nearest_future_expiry(expiries: list, now: datetime):
    """Pick the nearest upcoming expiry from a list of dates, rolling past
    today's expiry once the session has closed."""
    today = now.date()
    future = sorted(d for d in expiries if d >= today)
    if not future:
        return None
    nearest = future[0]
    if nearest == today:
        mkt_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        if now > mkt_close and len(future) > 1:
            nearest = future[1]
    return nearest


def get_next_weekly_expiry() -> datetime:
    """
    Get the next NIFTY weekly expiry. Source priority:
      1. AngelOne authenticated searchScrip API (live, authoritative)
      2. Public NFO scrip master file (when reachable)
      3. Last known expiry cached this session
      4. Calculated next weekly expiry weekday (Tuesday) as last resort
    Returns a tz-aware datetime, or None so the UI shows '---'.
    """
    now = datetime.now(IST)
    today = now.date()

    # 1) Authoritative: ask AngelOne directly (works even if scrip master 404s)
    try:
        ao_expiries = _get_expiry_from_angelone()
        nearest = _nearest_future_expiry(ao_expiries, now)
        if nearest:
            expiry_dt = datetime.combine(nearest, datetime.min.time()).replace(tzinfo=IST)
            st.session_state["_last_known_expiry"] = expiry_dt
            return expiry_dt
    except Exception as e:
        logger.debug(f"AngelOne expiry discovery failed: {e}")

    # 2) Public NFO scrip master
    try:
        nearest = _nearest_future_expiry(_load_nifty_expiries(), now)
        if nearest:
            expiry_dt = datetime.combine(nearest, datetime.min.time()).replace(tzinfo=IST)
            st.session_state["_last_known_expiry"] = expiry_dt
            return expiry_dt
    except Exception as e:
        logger.debug(f"Expiry fetch from scrip master failed: {e}")

    # 3) Last known expiry from session_state if still valid
    cached = st.session_state.get("_last_known_expiry")
    if cached is not None:
        cached_date = cached.date() if hasattr(cached, "date") else cached
        if cached_date >= today:
            return cached

    # 4) Last resort: calculate next weekly expiry weekday (Tuesday)
    try:
        next_exp = _calc_next_expiry_day()
        return datetime.combine(next_exp, datetime.min.time()).replace(tzinfo=IST)
    except Exception:
        pass

    return None


def get_expiry_string(expiry_dt) -> str:
    """Format expiry date as AngelOne API expects, e.g. '27JUN2024'. Returns '---' if None."""
    if expiry_dt is None:
        return "---"
    return expiry_dt.strftime("%d%b%Y").upper()


def get_expiry_countdown(expiry_dt) -> str:
    """Return human-readable countdown string to expiry. Returns '---' if None."""
    if expiry_dt is None:
        return "---"
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


# AngelOne getCandleData max calendar days per request, by interval.
# (FIVE_MINUTE = 100 days; we chunk conservatively below these caps.)
_CANDLE_MAX_DAYS = {1: 25, 2: 50, 3: 50, 5: 90, 10: 90, 15: 180, 30: 180, 60: 350}


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_candle_range(interval_minutes: int, from_date_str: str,
                       to_date_str: str) -> pd.DataFrame:
    """
    Fetch OHLCV candles for an arbitrary date range, chunked to respect
    AngelOne's per-request day caps (e.g. 100 days for 5-min). Dates are
    'YYYY-MM-DD' strings (inclusive). Returns one combined, de-duplicated,
    time-sorted DataFrame. Empty DataFrame if not connected.
    """
    obj = get_client()
    if obj is None:
        return _EMPTY_CANDLES.copy()

    try:
        interval_str = INTERVAL_MAP.get(interval_minutes, "FIVE_MINUTE")
        start = datetime.strptime(from_date_str, "%Y-%m-%d")
        end = datetime.strptime(to_date_str, "%Y-%m-%d")
        if start > end:
            start, end = end, start
        max_days = _CANDLE_MAX_DAYS.get(interval_minutes, 90)

        frames = []
        cur = start
        while cur <= end:
            chunk_end = min(cur + timedelta(days=max_days - 1), end)
            params = {
                "exchange": NIFTY_EXCHANGE,
                "symboltoken": NIFTY_TOKEN,
                "interval": interval_str,
                "fromdate": cur.strftime("%Y-%m-%d") + " 09:15",
                "todate": chunk_end.strftime("%Y-%m-%d") + " 15:30",
            }
            try:
                resp = obj.getCandleData(params)
                if resp and resp.get("status") and resp.get("data"):
                    chunk = pd.DataFrame(
                        resp["data"],
                        columns=["timestamp", "open", "high", "low", "close", "volume"],
                    )
                    frames.append(chunk)
            except Exception as e:
                logger.warning(f"Candle chunk {cur}–{chunk_end} failed: {e}")
            cur = chunk_end + timedelta(days=1)

        if not frames:
            return _EMPTY_CANDLES.copy()

        df = pd.concat(frames, ignore_index=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(inplace=True)
        df.drop_duplicates(subset=["timestamp"], inplace=True)
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    except Exception as e:
        logger.error(f"fetch_candle_range error: {e}")
        return _EMPTY_CANDLES.copy()


# ── NSE expiry-day weekday timeline ──────────────────────────────────────────
# NIFTY index option expiry weekday across history (Mon=0 … Sun=6):
#   • Since inception (monthly last-Thursday; weekly Thursdays from Feb 2019)
#     the expiry weekday was THURSDAY (3).
#   • SEBI circular (26-May-2025) standardised equity-derivative expiries to
#     Tuesday/Thursday. NSE moved NIFTY expiry to TUESDAY (1) from 01-Sep-2025.
# To add a future change, append (effective_date, weekday) — kept sorted.
from datetime import date as _date

_DEFAULT_EXPIRY_WEEKDAY = 3  # Thursday
_EXPIRY_RULE_TIMELINE = [
    (_date(2025, 9, 1), 1),  # NSE → Tuesday, effective 1 Sep 2025
]


def expiry_weekday_for(d) -> int:
    """Return the NSE NIFTY expiry weekday rule in effect on date `d`."""
    if hasattr(d, "date"):
        d = d.date()
    wd = _DEFAULT_EXPIRY_WEEKDAY
    for eff, w in sorted(_EXPIRY_RULE_TIMELINE):
        if d >= eff:
            wd = w
    return wd


def get_expiry_timeline_summary() -> pd.DataFrame:
    """Human-readable table of when the NIFTY expiry weekday changed."""
    wd_name = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
               4: "Friday", 5: "Saturday", 6: "Sunday"}
    rows = [{
        "From": "Inception (weekly: 11 Feb 2019)",
        "To": "31 Aug 2025",
        "Expiry Day": "Thursday",
        "Note": "Monthly = last Thursday; weekly Thursdays from Feb 2019",
    }]
    prev_to = "Ongoing"
    for eff, w in sorted(_EXPIRY_RULE_TIMELINE):
        rows.append({
            "From": eff.strftime("%d %b %Y"),
            "To": prev_to,
            "Expiry Day": wd_name[w],
            "Note": "SEBI 26-May-2025 circular; NSE→Tuesday to avoid BSE clash",
        })
    return pd.DataFrame(rows)


def flag_expiry_days(dates: pd.Series) -> pd.Series:
    """
    Given a Series of (datetime) trading dates, return a boolean Series marking
    which are NIFTY expiry days. Within each ISO week the expiry is the latest
    trading day whose weekday ≤ the rule weekday (handles holiday roll-back),
    using the weekday rule in effect that week.
    """
    s = pd.to_datetime(dates).reset_index(drop=True)
    flags = pd.Series(False, index=s.index)
    iso = s.dt.isocalendar()
    wd = s.dt.weekday
    grp = pd.DataFrame({"iy": iso["year"].values, "iw": iso["week"].values,
                        "wd": wd.values, "date": s.values})
    for (_, _), block in grp.groupby(["iy", "iw"]):
        rule = expiry_weekday_for(pd.Timestamp(block["date"].iloc[-1]))
        cand = block[block["wd"] <= rule]
        if cand.empty:
            continue
        # latest trading day with weekday <= rule weekday
        pick = cand.loc[cand["wd"].idxmax()]
        flags.loc[pick.name] = True
    return flags


# Scrip master URLs to try in order. The correct filename is
# OpenAPIScripMaster.json (NOT ...SymbolMaster.json — that 404s).
_SCRIP_MASTER_URLS = [
    "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json",
    "https://margincalculator.angelone.in/OpenAPI_File/files/OpenAPIScripMaster.json",
]

# Module-level cache: only stores successful non-empty results.
# Falls back to last good copy on network failure — no hour-long blackouts.
_MASTER_CACHE: dict = {"df": None, "ts": 0.0, "error": "", "stale": None}
_CHAIN_DIAG: dict = {"msg": ""}   # last options-chain fetch diagnostic


def _load_nifty_master_raw() -> pd.DataFrame:
    """
    Download the NFO scrip master (~15-20 MB, 100k+ rows) trying each URL in
    _SCRIP_MASTER_URLS. Caches at module level for 1 hour; falls back to the
    last successful copy so the UI never goes dark on a transient failure.
    """
    import requests

    now = time.time()
    if _MASTER_CACHE["df"] is not None and (now - _MASTER_CACHE["ts"]) < 3600:
        return _MASTER_CACHE["df"]

    last_err = ""
    for url in _SCRIP_MASTER_URLS:
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            last_err = ""
            break
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            logger.warning(f"Scrip master URL {url} failed: {e}")
            data = None

    if data is None:
        _MASTER_CACHE["error"] = f"Scrip master download failed: {last_err}"
        stale = _MASTER_CACHE["stale"]
        return stale if stale is not None else pd.DataFrame()

    rows = []
    for item in data:
        if item.get("name") != "NIFTY" or item.get("instrumenttype") != "OPTIDX":
            continue
        symbol = item.get("symbol", "")
        opt_type = "CE" if symbol.endswith("CE") else "PE" if symbol.endswith("PE") else None
        if opt_type is None:
            continue
        exp_raw = str(item.get("expiry", "")).upper()
        try:
            exp_date = datetime.strptime(exp_raw, "%d%b%Y").date()
        except Exception:
            continue
        try:
            strike = float(item.get("strike", 0)) / 100.0  # master strike is in paise
        except Exception:
            continue
        rows.append({
            "strike": strike,
            "option_type": opt_type,
            "token": str(item.get("token", "")),
            "symbol": symbol,
            "expiry": exp_raw,
            "expiry_date": exp_date,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        _MASTER_CACHE["df"] = df
        _MASTER_CACHE["ts"] = now
        _MASTER_CACHE["stale"] = df
        _MASTER_CACHE["error"] = ""
    else:
        _MASTER_CACHE["error"] = (
            "Scrip master downloaded but contained no NIFTY OPTIDX rows."
        )
    return df


# NSE weekly expiry weekday for NIFTY. NSE moved NIFTY weekly expiry to
# TUESDAY (weekday=1). Change this single constant if NSE shifts it again.
NIFTY_EXPIRY_WEEKDAY = 1  # Monday=0, Tuesday=1, ... Sunday=6


def _calc_next_expiry_day():
    """
    Calculate the next NSE weekly expiry day (Tuesday) from today.
    Used as fallback when the scrip master is unavailable. Returns a date.
    """
    now = datetime.now(IST)
    today = now.date()
    days_ahead = (NIFTY_EXPIRY_WEEKDAY - today.weekday()) % 7
    if days_ahead == 0:
        # If today IS expiry day, roll forward only after the session closes
        mkt_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        if now > mkt_close:
            days_ahead = 7
    return today + timedelta(days=days_ahead)


def get_options_diagnostic() -> str:
    """Return a human-readable reason why options data may be unavailable."""
    if not is_connected():
        return "Not connected to AngelOne — click Connect to load live option data."
    if _MASTER_CACHE["error"]:
        return _MASTER_CACHE["error"]
    return _CHAIN_DIAG.get("msg", "")


def _load_nifty_expiries() -> list:
    """Return sorted unique NIFTY option expiry dates from the scrip master."""
    raw = _load_nifty_master_raw()
    if raw.empty:
        return []
    return sorted(set(raw["expiry_date"].tolist()))


def _load_nifty_option_master(expiry_str: str) -> pd.DataFrame:
    """
    Return NIFTY index options for the given expiry from the cached master.
    Columns: strike, option_type (CE/PE), token, symbol.
    """
    raw = _load_nifty_master_raw()
    if raw.empty:
        return pd.DataFrame()
    sub = raw[raw["expiry"] == expiry_str.upper()]
    return sub[["strike", "option_type", "token", "symbol"]].reset_index(drop=True)


def _chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _parse_expiry_from_symbol(ts: str):
    """Extract the expiry date from an NFO NIFTY option tradingsymbol.

    e.g. 'NIFTY16JUN2623600CE' → date(2026, 6, 16). Returns None on failure.
    """
    import re
    if not ts or not ts.startswith("NIFTY") or ts.startswith("NIFTYNXT"):
        return None
    m = re.match(r"^NIFTY(\d{2}[A-Z]{3}\d{2})\d+(?:CE|PE)$", ts)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%d%b%y").date()
    except Exception:
        return None


def _get_expiry_from_angelone() -> list:
    """
    Discover NIFTY weekly expiry dates directly from AngelOne via the
    authenticated searchScrip API (works when the public scrip master 404s).
    Returns a sorted list of unique expiry dates, or [] on failure.
    Result is cached at module level for 30 minutes.
    """
    now = time.time()
    cache = _MASTER_CACHE.get("ao_expiries")
    cache_ts = _MASTER_CACHE.get("ao_expiries_ts", 0)
    if cache and (now - cache_ts) < 1800:
        return cache

    obj = get_client()
    if obj is None:
        return []
    try:
        res = obj.searchScrip("NFO", "NIFTY")
    except Exception as e:
        logger.warning(f"searchScrip(NIFTY) for expiry discovery failed: {e}")
        return []
    if not (res and res.get("status") and res.get("data")):
        return []

    dates = set()
    for item in res["data"]:
        ts = str(item.get("tradingsymbol", "")).upper()
        d = _parse_expiry_from_symbol(ts)
        if d is not None:
            dates.add(d)
    out = sorted(dates)
    if out:
        _MASTER_CACHE["ao_expiries"] = out
        _MASTER_CACHE["ao_expiries_ts"] = now
    return out


def _parse_option_symbol(ts: str):
    """
    Parse an NFO NIFTY option tradingsymbol into (strike, option_type).
    Accepts forms like 'NIFTY16JUN2623600CE'. Returns (None, None) if it is
    not a plain NIFTY index option (e.g. FINNIFTY / NIFTYNXT50 / futures).
    """
    import re
    if not ts or not ts.startswith("NIFTY"):
        return None, None
    # Exclude FINNIFTY (starts FIN), BANKNIFTY (BANK), MIDCPNIFTY, NIFTYNXT50
    if ts.startswith("NIFTYNXT"):
        return None, None
    if not ts.endswith(("CE", "PE")):
        return None, None
    opt_type = ts[-2:]
    # The strike is the trailing run of digits immediately before CE/PE
    m = re.search(r"(\d+)(CE|PE)$", ts)
    if not m:
        return None, None
    try:
        strike = float(m.group(1))
    except Exception:
        return None, None
    # Sanity: NIFTY strikes are 4-6 digit whole numbers (e.g. 23600)
    if strike < 1000 or strike > 100000:
        return None, None
    return strike, opt_type


def _search_nifty_options(obj, expiry_dt, spot: float, n: int = 12) -> pd.DataFrame:
    """
    Fallback option-contract discovery using the AUTHENTICATED searchScrip API
    (works even when the public scrip-master file 404s). Searches by the
    compact expiry prefix, parses the returned tradingsymbols, and keeps
    ATM ± n strikes. Returns columns: strike, option_type, token, symbol.
    """
    if obj is None or expiry_dt is None:
        return pd.DataFrame()

    # Build candidate expiry prefixes AngelOne may use, e.g. 16JUN26 / 16JUN2026
    yy = expiry_dt.strftime("%y")
    yyyy = expiry_dt.strftime("%Y")
    ddmon = expiry_dt.strftime("%d%b").upper()
    prefixes = [f"NIFTY{ddmon}{yy}", f"NIFTY{ddmon}{yyyy}"]

    rows = {}
    for q in prefixes:
        try:
            res = obj.searchScrip("NFO", q)
        except Exception as e:
            logger.warning(f"searchScrip({q}) failed: {e}")
            continue
        if not (res and res.get("status") and res.get("data")):
            continue
        for item in res["data"]:
            ts = str(item.get("tradingsymbol", "")).upper()
            tok = str(item.get("symboltoken", ""))
            strike, opt_type = _parse_option_symbol(ts)
            if strike is None or not tok:
                continue
            rows[(strike, opt_type)] = {
                "strike": strike, "option_type": opt_type,
                "token": tok, "symbol": ts,
            }
        if rows:
            break  # got results from this prefix; no need to try the next

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(list(rows.values()))
    # Keep ATM ± n strikes
    all_strikes = sorted(df["strike"].unique())
    if spot and spot > 0 and all_strikes:
        atm = min(all_strikes, key=lambda s: abs(s - spot))
        atm_idx = all_strikes.index(atm)
        lo = max(0, atm_idx - n)
        hi = min(len(all_strikes), atm_idx + n + 1)
        keep = set(all_strikes[lo:hi])
        df = df[df["strike"].isin(keep)]
    return df.reset_index(drop=True)


@st.cache_data(ttl=10)
def fetch_options_chain(expiry_str: str = None) -> pd.DataFrame:
    """
    Fetch the NIFTY options chain (ATM ±12 strikes) from AngelOne:
      • getMarketData(FULL) → OI, volume, LTP, best bid/ask per strike
      • Black-Scholes (local) → delta, gamma, theta, vega, IV from each LTP
    Returns one row per strike with ce_*/pe_* columns.
    Returns an empty DataFrame if not connected — no simulated data.
    """
    try:
        _CHAIN_DIAG["msg"] = ""
        if expiry_str is None or expiry_str == "---":
            expiry_dt = get_next_weekly_expiry()
            expiry_str = get_expiry_string(expiry_dt)
        if expiry_str == "---":
            _CHAIN_DIAG["msg"] = (
                "Expiry unavailable (scrip master not loaded) — cannot map option tokens."
            )
            return pd.DataFrame()

        obj = get_client()
        if obj is None:
            return pd.DataFrame()

        spot = fetch_ltp()

        # Primary: public scrip master. Fallback: authenticated searchScrip API
        # (the public scrip-master file currently 404s, but searchScrip works).
        master = _load_nifty_option_master(expiry_str)
        if master.empty:
            expiry_dt = get_next_weekly_expiry()
            master = _search_nifty_options(obj, expiry_dt, spot, n=12)
            if master.empty:
                _CHAIN_DIAG["msg"] = (
                    f"No NIFTY option contracts found for {expiry_str} via scrip master "
                    "or searchScrip API."
                )
                logger.error(f"No NIFTY options found for expiry {expiry_str}")
                return pd.DataFrame()

        # Limit to ATM ±12 strikes to stay within the 50-token market-data cap
        all_strikes = sorted(master["strike"].unique())
        if spot and spot > 0:
            atm = min(all_strikes, key=lambda s: abs(s - spot))
            atm_idx = all_strikes.index(atm)
            lo = max(0, atm_idx - 12)
            hi = min(len(all_strikes), atm_idx + 13)
            keep = set(all_strikes[lo:hi])
            master = master[master["strike"].isin(keep)]

        token_to_meta = {
            r["token"]: (r["strike"], r["option_type"])
            for _, r in master.iterrows()
        }
        tokens = list(token_to_meta.keys())

        # ── Market data (OI / volume / LTP / depth) ──────────────────────────
        md_by_token = {}
        for batch in _chunked(tokens, 50):
            try:
                md = obj.getMarketData("FULL", {"NFO": batch})
                if md and md.get("status") and md.get("data"):
                    data_payload = md["data"]
                    # API may return a dict {"fetched": [...]} or a list directly
                    if isinstance(data_payload, dict):
                        fetched = data_payload.get("fetched") or []
                    elif isinstance(data_payload, list):
                        fetched = data_payload
                    else:
                        fetched = []
                    for item in fetched:
                        if not isinstance(item, dict):
                            continue
                        # Handle both camelCase and lowercase key variants
                        tok = str(
                            item.get("symbolToken")
                            or item.get("symboltoken")
                            or item.get("token")
                            or ""
                        )
                        if tok:
                            md_by_token[tok] = item
            except Exception as e:
                logger.error(f"getMarketData error: {e}")

        if not md_by_token:
            _CHAIN_DIAG["msg"] = (
                "getMarketData returned no rows for the option tokens "
                "(API limit, session, or NFO subscription)."
            )

        def _best_depth(item, side):
            try:
                lvls = item.get("depth", {}).get(side, [])
                return float(lvls[0].get("price", 0)) if lvls else 0.0
            except Exception:
                return 0.0

        # ── Assemble per-strike rows (OI / volume / LTP / depth) ─────────────
        strikes = sorted(master["strike"].unique())
        rows = []
        for strike in strikes:
            row = {"strike": float(strike)}
            for opt in ("ce", "pe"):
                ot = opt.upper()
                sub = master[(master["strike"] == strike) & (master["option_type"] == ot)]
                md = md_by_token.get(sub["token"].iloc[0]) if not sub.empty else None
                if md:
                    row[f"{opt}_oi"] = int(float(md.get("opnInterest", 0) or 0))
                    row[f"{opt}_volume"] = int(float(md.get("tradeVolume", 0) or 0))
                    row[f"{opt}_ltp"] = float(md.get("ltp", 0) or 0)
                    row[f"{opt}_bid"] = _best_depth(md, "buy")
                    row[f"{opt}_ask"] = _best_depth(md, "sell")
                else:
                    row[f"{opt}_oi"] = 0
                    row[f"{opt}_volume"] = 0
                    row[f"{opt}_ltp"] = 0.0
                    row[f"{opt}_bid"] = 0.0
                    row[f"{opt}_ask"] = 0.0
                # Greek columns are filled below via Black-Scholes
                row[f"{opt}_iv"] = 0.0
                row[f"{opt}_delta"] = 0.0
                row[f"{opt}_gamma"] = 0.0
                row[f"{opt}_theta"] = 0.0
                row[f"{opt}_vega"] = 0.0
            rows.append(row)

        df = pd.DataFrame(rows)
        df.sort_values("strike", inplace=True)
        df.reset_index(drop=True, inplace=True)

        # ── Greeks via Black-Scholes (computed locally from option LTP) ──────
        # More reliable than the optionGreek API, which often returns zeros.
        try:
            from modules.black_scholes import compute_chain_greeks
            expiry_dt = get_next_weekly_expiry()
            df = compute_chain_greeks(df, spot, expiry_dt, datetime.now(IST))
        except Exception as e:
            logger.error(f"Black-Scholes greeks error: {e}")

        if not df.empty:
            # Persist for fallback when market is closed / chain temporarily unavailable
            st.session_state["_last_options_df"] = df
            st.session_state["_last_options_expiry"] = expiry_str
        return df

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
