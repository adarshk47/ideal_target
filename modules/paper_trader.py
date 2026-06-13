"""
Paper Trading Module for Nifty50.
Auto-generates and tracks paper trades based on pattern signals.
Only active during market hours.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import pytz
from typing import List, Dict, Optional

IST = pytz.timezone("Asia/Kolkata")


def is_market_open() -> bool:
    now = datetime.now(IST)
    if now.weekday() >= 5:
        return False
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_open <= now <= market_close


def init_paper_trades():
    if "paper_trades" not in st.session_state:
        st.session_state["paper_trades"] = []
    if "paper_trade_counter" not in st.session_state:
        st.session_state["paper_trade_counter"] = 0


def add_paper_trade(signal, pattern_name: str, spot_price: float, source: str = "AUTO"):
    """Add a new paper trade from a pattern signal."""
    init_paper_trades()
    now = datetime.now(IST)
    st.session_state["paper_trade_counter"] += 1
    trade_id = st.session_state["paper_trade_counter"]

    # Determine option strike and type
    atm = round(spot_price / 50) * 50
    if signal.signal == "BUY":
        option_type = "CE"
        strike = atm  # Buy ATM call on bullish signal
    else:
        option_type = "PE"
        strike = atm  # Buy ATM put on bearish signal

    trade = {
        "id": trade_id,
        "time": now.strftime("%H:%M:%S"),
        "date": now.strftime("%d-%b-%Y"),
        "pattern": pattern_name,
        "signal": signal.signal,
        "option": f"NIFTY {int(strike)} {option_type}",
        "entry_spot": round(spot_price, 2),
        "entry": round(signal.entry, 2),
        "stop_loss": round(signal.stop_loss, 2),
        "target": round(signal.target, 2),
        "rr": signal.risk_reward,
        "status": "OPEN",
        "exit_price": None,
        "exit_time": None,
        "pnl": 0.0,
        "pnl_pct": 0.0,
        "source": source,
        "confidence": getattr(signal, "confidence", "MEDIUM"),
    }
    st.session_state["paper_trades"].append(trade)
    return trade_id


def update_paper_trades(current_spot: float):
    """Check open trades and mark them complete if SL or target is hit."""
    init_paper_trades()
    if not st.session_state["paper_trades"]:
        return

    now = datetime.now(IST)
    for trade in st.session_state["paper_trades"]:
        if trade["status"] != "OPEN":
            continue

        entry = trade["entry"]
        sl = trade["stop_loss"]
        target = trade["target"]
        signal = trade["signal"]

        if signal == "BUY":
            if current_spot <= sl:
                trade["status"] = "LOSS"
                trade["exit_price"] = round(sl, 2)
                trade["exit_time"] = now.strftime("%H:%M:%S")
                trade["pnl"] = round(sl - entry, 2)
                trade["pnl_pct"] = round((sl - entry) / entry * 100, 2)
            elif current_spot >= target:
                trade["status"] = "PROFIT"
                trade["exit_price"] = round(target, 2)
                trade["exit_time"] = now.strftime("%H:%M:%S")
                trade["pnl"] = round(target - entry, 2)
                trade["pnl_pct"] = round((target - entry) / entry * 100, 2)
        else:  # SELL
            if current_spot >= sl:
                trade["status"] = "LOSS"
                trade["exit_price"] = round(sl, 2)
                trade["exit_time"] = now.strftime("%H:%M:%S")
                trade["pnl"] = round(entry - sl, 2)
                trade["pnl_pct"] = round((entry - sl) / entry * 100, 2)
            elif current_spot <= target:
                trade["status"] = "PROFIT"
                trade["exit_price"] = round(target, 2)
                trade["exit_time"] = now.strftime("%H:%M:%S")
                trade["pnl"] = round(entry - target, 2)
                trade["pnl_pct"] = round((entry - target) / entry * 100, 2)


def get_trades_df() -> pd.DataFrame:
    init_paper_trades()
    if not st.session_state["paper_trades"]:
        return pd.DataFrame()
    return pd.DataFrame(st.session_state["paper_trades"])


def get_paper_trade_summary() -> Dict:
    df = get_trades_df()
    if df.empty:
        return {"total": 0, "open": 0, "profit": 0, "loss": 0, "total_pnl": 0.0, "win_rate": 0.0}

    closed = df[df["status"] != "OPEN"]
    profits = (df["status"] == "PROFIT").sum()
    losses = (df["status"] == "LOSS").sum()
    total_closed = profits + losses
    win_rate = profits / total_closed * 100 if total_closed > 0 else 0

    return {
        "total": len(df),
        "open": (df["status"] == "OPEN").sum(),
        "profit": int(profits),
        "loss": int(losses),
        "total_pnl": round(df["pnl"].sum(), 2),
        "win_rate": round(win_rate, 1),
    }


def should_add_new_trade(pattern_name: str, signal_type: str) -> bool:
    """Avoid duplicate trades for same pattern in short window."""
    init_paper_trades()
    recent = [
        t for t in st.session_state["paper_trades"]
        if t["pattern"] == pattern_name
        and t["signal"] == signal_type
        and t["status"] == "OPEN"
    ]
    return len(recent) == 0


def clear_all_trades():
    st.session_state["paper_trades"] = []
    st.session_state["paper_trade_counter"] = 0
