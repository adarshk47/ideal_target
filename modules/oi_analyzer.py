"""
OI (Open Interest) Analyzer Module
Calculates Delta OI, net OI trends, and generates directional arrows for chart annotations.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import streamlit as st
import logging

logger = logging.getLogger(__name__)

IST = pytz.timezone("Asia/Kolkata")

TIMEFRAMES = [1, 2, 5, 10, 15, 30, 60]


@st.cache_data(ttl=5)
def get_oi_snapshot() -> dict:
    """
    Get current options chain OI snapshot.
    Returns dict with strike -> {ce_oi, pe_oi, ce_volume, pe_volume, net_oi}.
    """
    from modules.angelone_client import fetch_options_chain, get_next_weekly_expiry, get_expiry_string
    expiry_dt = get_next_weekly_expiry()
    expiry_str = get_expiry_string(expiry_dt)
    df = fetch_options_chain(expiry_str)

    if df.empty:
        return {}

    snapshot = {}
    for _, row in df.iterrows():
        strike = int(row["strike"])
        snapshot[strike] = {
            "ce_oi": int(row.get("ce_oi", 0)),
            "pe_oi": int(row.get("pe_oi", 0)),
            "ce_volume": int(row.get("ce_volume", 0)),
            "pe_volume": int(row.get("pe_volume", 0)),
            "net_oi": int(row.get("pe_oi", 0)) - int(row.get("ce_oi", 0)),
        }
    return snapshot


def store_oi_snapshot(snapshot: dict):
    """Store OI snapshot with timestamp in session_state history."""
    if "oi_history" not in st.session_state:
        st.session_state["oi_history"] = []

    entry = {
        "timestamp": datetime.now(IST),
        "snapshot": snapshot,
    }
    st.session_state["oi_history"].append(entry)

    # Keep only last 2 hours of data
    cutoff = datetime.now(IST) - timedelta(hours=2)
    st.session_state["oi_history"] = [
        e for e in st.session_state["oi_history"]
        if e["timestamp"] > cutoff
    ]


def get_delta_oi_for_timeframe(minutes: int) -> dict:
    """
    Calculate delta OI (change in net OI) over given timeframe in minutes.
    Returns dict with: delta_net_oi, direction ("BULLISH"/"BEARISH"/"NEUTRAL"),
    ce_oi_change, pe_oi_change, arrow, color.
    """
    history = st.session_state.get("oi_history", [])
    if len(history) < 2:
        return _neutral_delta()

    now = datetime.now(IST)
    cutoff = now - timedelta(minutes=minutes)

    # Find closest snapshot to cutoff time
    past_entries = [e for e in history if e["timestamp"] <= cutoff]
    if not past_entries:
        # Not enough history, use oldest available
        past_entries = [history[0]]

    past_snap = past_entries[-1]["snapshot"]
    current_snap = history[-1]["snapshot"]

    # Calculate aggregate changes
    total_ce_oi_change = 0
    total_pe_oi_change = 0
    total_net_oi_change = 0
    strikes_analyzed = 0

    for strike in current_snap:
        if strike in past_snap:
            ce_change = current_snap[strike]["ce_oi"] - past_snap[strike]["ce_oi"]
            pe_change = current_snap[strike]["pe_oi"] - past_snap[strike]["pe_oi"]
            net_change = pe_change - ce_change
            total_ce_oi_change += ce_change
            total_pe_oi_change += pe_change
            total_net_oi_change += net_change
            strikes_analyzed += 1

    if strikes_analyzed == 0:
        return _neutral_delta()

    # Determine direction
    # Rising net OI (more PE writing than CE) = BEARISH
    # Falling net OI (more CE writing than PE) = BULLISH
    threshold = 50000  # Minimum change to consider significant

    if total_net_oi_change > threshold:
        direction = "BEARISH"
        arrow = "↓"
        color = "#ef4444"
    elif total_net_oi_change < -threshold:
        direction = "BULLISH"
        arrow = "↑"
        color = "#22c55e"
    else:
        direction = "NEUTRAL"
        arrow = "→"
        color = "#f59e0b"

    return {
        "direction": direction,
        "arrow": arrow,
        "color": color,
        "delta_net_oi": total_net_oi_change,
        "ce_oi_change": total_ce_oi_change,
        "pe_oi_change": total_pe_oi_change,
        "timeframe_minutes": minutes,
        "strikes_analyzed": strikes_analyzed,
    }


def _neutral_delta() -> dict:
    return {
        "direction": "NEUTRAL",
        "arrow": "→",
        "color": "#f59e0b",
        "delta_net_oi": 0,
        "ce_oi_change": 0,
        "pe_oi_change": 0,
        "timeframe_minutes": 0,
        "strikes_analyzed": 0,
    }


def get_all_timeframe_deltas() -> dict:
    """Get delta OI for all standard timeframes."""
    results = {}
    for tf in TIMEFRAMES:
        results[tf] = get_delta_oi_for_timeframe(tf)
    return results


def compute_oi_delta_bars(df_candles: pd.DataFrame, options_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a per-bar OI delta series aligned to the candle timestamps.
    Used for the OI subplot below the main chart.
    """
    if df_candles.empty or options_df.empty:
        return pd.DataFrame()

    # Aggregate total net OI from options chain (snapshot)
    total_ce_oi = options_df["ce_oi"].sum()
    total_pe_oi = options_df["pe_oi"].sum()
    net_oi = total_pe_oi - total_ce_oi

    # Simulate bar-by-bar OI delta using volume as proxy
    history = st.session_state.get("oi_history", [])

    # Build a time-series of net OI changes aligned to candles
    timestamps = df_candles["timestamp"].tolist()
    oi_values = []

    np.random.seed(42)
    # If we have real history, use it; otherwise simulate
    base_net_oi = net_oi
    for i, ts in enumerate(timestamps):
        if history:
            # Find closest historical snapshot
            matching = [
                e for e in history
                if abs((e["timestamp"].replace(tzinfo=None) - pd.Timestamp(ts).to_pydatetime()).total_seconds()) < 300
            ]
            if matching:
                snap = matching[-1]["snapshot"]
                bar_net = sum(v["pe_oi"] - v["ce_oi"] for v in snap.values())
            else:
                noise = np.random.normal(0, base_net_oi * 0.002)
                bar_net = base_net_oi + noise * i
        else:
            # Mock OI delta: fluctuates around current net
            noise = np.random.normal(0, abs(base_net_oi) * 0.001 + 10000)
            bar_net = base_net_oi + noise

        oi_values.append(bar_net)

    oi_df = pd.DataFrame({
        "timestamp": timestamps,
        "net_oi": oi_values,
    })

    # Calculate bar-by-bar delta
    oi_df["delta_oi"] = oi_df["net_oi"].diff().fillna(0)
    oi_df["color"] = oi_df["delta_oi"].apply(
        lambda x: "#22c55e" if x < 0 else "#ef4444"  # negative delta = bullish
    )
    return oi_df


def get_most_traded_strikes(options_df: pd.DataFrame, spot_price: float, n_strikes: int = 5, top_n: int = 10) -> pd.DataFrame:
    """
    Get the most traded strike prices (by volume) within ATM ± n_strikes.
    Returns sorted DataFrame.
    """
    from modules.angelone_client import get_strike_range
    if options_df.empty:
        return pd.DataFrame()

    strike_range = get_strike_range(spot_price, n_strikes)
    filtered = options_df[options_df["strike"].isin(strike_range)].copy()

    if filtered.empty:
        return filtered

    filtered["total_volume"] = filtered["ce_volume"] + filtered["pe_volume"]
    filtered["ce_pe_ratio"] = (filtered["ce_oi"] / filtered["pe_oi"].replace(0, 1)).round(2)
    filtered["pcr"] = (filtered["pe_oi"] / filtered["ce_oi"].replace(0, 1)).round(2)
    filtered["net_oi"] = filtered["pe_oi"] - filtered["ce_oi"]

    return filtered.sort_values("total_volume", ascending=False).reset_index(drop=True)


def get_oi_table_for_timeframes(options_df: pd.DataFrame, spot_price: float) -> pd.DataFrame:
    """
    Build OI difference table showing CE vs PE trend for each timeframe.
    Returns DataFrame with columns: timeframe, ce_oi_change, pe_oi_change, delta_net_oi, direction, arrow.
    """
    rows = []
    deltas = get_all_timeframe_deltas()

    for tf in TIMEFRAMES:
        d = deltas[tf]
        rows.append({
            "Timeframe": f"{tf} min",
            "CE OI Change": d.get("ce_oi_change", 0),
            "PE OI Change": d.get("pe_oi_change", 0),
            "Net OI Δ": d.get("delta_net_oi", 0),
            "Direction": d.get("direction", "NEUTRAL"),
            "Arrow": d.get("arrow", "→"),
            "Color": d.get("color", "#f59e0b"),
        })

    return pd.DataFrame(rows)


def get_volume_by_timeframe(options_df: pd.DataFrame, spot_price: float, n_strikes: int = 5) -> dict:
    """
    Get volume breakdown by timeframe for most traded strikes.
    Since we don't have historical volume per-bar from the API, we estimate
    using current snapshot volumes scaled by time proportion.
    """
    from modules.angelone_client import get_strike_range
    strike_range = get_strike_range(spot_price, n_strikes)
    filtered = options_df[options_df["strike"].isin(strike_range)].copy()

    result = {}
    for tf in TIMEFRAMES:
        if filtered.empty:
            result[tf] = pd.DataFrame()
            continue

        # Scale volume by timeframe proportion (rough estimate)
        scale = tf / 375.0  # 375 minutes in trading day
        tf_df = filtered.copy()
        tf_df["est_ce_volume"] = (tf_df["ce_volume"] * scale).astype(int)
        tf_df["est_pe_volume"] = (tf_df["pe_volume"] * scale).astype(int)
        tf_df["est_total_volume"] = tf_df["est_ce_volume"] + tf_df["est_pe_volume"]
        tf_df = tf_df.sort_values("est_total_volume", ascending=False)
        result[tf] = tf_df[["strike", "est_ce_volume", "est_pe_volume", "est_total_volume", "ce_oi", "pe_oi"]].reset_index(drop=True)

    return result
