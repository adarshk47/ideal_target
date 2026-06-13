"""
Paper Trading Module for Nifty50, Sensex, SBIN.
Auto-generates and tracks paper trades based on pattern signals.
Works both during live market AND in post-market simulation mode.
All state is keyed per-instrument so NIFTY/SENSEX/SBIN trades never mix.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import pytz
from typing import Dict

IST = pytz.timezone("Asia/Kolkata")


def is_market_open() -> bool:
    now = datetime.now(IST)
    if now.weekday() >= 5:
        return False
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_open <= now <= market_close


def init_paper_trades(instrument: str = "NIFTY"):
    k = instrument
    if f"paper_trades_{k}" not in st.session_state:
        st.session_state[f"paper_trades_{k}"] = []
    if f"paper_trade_counter_{k}" not in st.session_state:
        st.session_state[f"paper_trade_counter_{k}"] = 0
    if f"chart_rec_trades_{k}" not in st.session_state:
        st.session_state[f"chart_rec_trades_{k}"] = []


def add_paper_trade(signal, pattern_name: str, spot_price: float,
                    source: str = "AUTO", simulated: bool = False,
                    option_ltp: float = 0.0, exit_info: dict = None,
                    entry_time: str = None, strike: float = None,
                    option_sl: float = None, option_target: float = None,
                    instrument: str = "NIFTY"):
    """Add a new paper trade from a pattern signal.

    option_ltp: if > 1, use as option premium entry price.
    option_sl / option_target: option premiums at the spot stop-loss / target
                levels (B-S priced on the entry-time ATM strike).
    exit_info:  {'status': 'PROFIT'|'LOSS', 'exit_time': str|None,
                 'exit_option_price': float|None}
    entry_time: 'HH:MM:SS' of the pattern candle.
    strike:     explicit ATM strike for the option name.
    instrument: 'NIFTY', 'SENSEX', or 'SBIN'.
    """
    init_paper_trades(instrument)
    now = datetime.now(IST)
    st.session_state[f"paper_trade_counter_{instrument}"] += 1
    trade_id = st.session_state[f"paper_trade_counter_{instrument}"]

    trade_time = entry_time if entry_time else now.strftime("%H:%M:%S")

    atm = int(strike) if strike else round(spot_price / 50) * 50
    option_type = "CE" if signal.signal == "BUY" else "PE"
    rr = float(signal.risk_reward or 1.5)

    use_option = bool(option_ltp and option_ltp > 1.0)
    if use_option:
        entry = round(float(option_ltp), 2)
        if option_sl is not None and option_target is not None:
            sl = round(float(option_sl), 2)
            target = round(float(option_target), 2)
        else:
            risk = round(entry * 0.25, 2)
            sl = round(entry - risk, 2)
            target = round(entry + risk * rr, 2)
        risk = abs(entry - sl)
    else:
        entry = round(float(signal.entry), 2)
        sl = round(float(signal.stop_loss), 2)
        target = round(float(signal.target), 2)
        risk = abs(entry - sl)

    status = "OPEN"
    exit_price = None
    exit_time = None
    pnl = 0.0
    pnl_pct = 0.0

    if simulated:
        if exit_info and exit_info.get("status") in ("PROFIT", "LOSS"):
            status = exit_info["status"]
            exit_time = exit_info.get("exit_time")
            exit_opt = exit_info.get("exit_option_price")
            if exit_opt is not None and use_option and exit_opt > 0:
                exit_price = exit_opt
                pnl = round(exit_price - entry, 2)
            elif status == "PROFIT":
                exit_price = target
                pnl = round(target - entry, 2)
            else:
                exit_price = sl
                pnl = round(sl - entry, 2)
            pnl_pct = round(pnl / entry * 100, 2) if entry else 0.0

        else:
            # R:R simulation fallback
            if use_option:
                sim_exit = entry + risk * (rr * 0.5)
                if sim_exit >= target:
                    status, exit_price = "PROFIT", target
                elif sim_exit <= sl:
                    status, exit_price = "LOSS", sl
                else:
                    status = "PROFIT" if rr >= 1.5 else "LOSS"
                    exit_price = round(sim_exit, 2)
                pnl = round(exit_price - entry, 2)
            else:
                if signal.signal == "BUY":
                    sim_exit = entry + risk * (rr * 0.5)
                    if sim_exit >= target:
                        status, exit_price = "PROFIT", target
                        pnl = round(target - entry, 2)
                    elif sim_exit <= sl:
                        status, exit_price = "LOSS", sl
                        pnl = round(sl - entry, 2)
                    else:
                        status = "PROFIT" if rr >= 1.5 else "LOSS"
                        exit_price = round(sim_exit, 2)
                        pnl = round(sim_exit - entry, 2)
                else:
                    sim_exit = entry - risk * (rr * 0.5)
                    if sim_exit <= target:
                        status, exit_price = "PROFIT", target
                        pnl = round(entry - target, 2)
                    elif sim_exit >= sl:
                        status, exit_price = "LOSS", sl
                        pnl = round(entry - sl, 2)
                    else:
                        status = "PROFIT" if rr >= 1.5 else "LOSS"
                        exit_price = round(sim_exit, 2)
                        pnl = round(entry - sim_exit, 2)

            pnl_pct = round(pnl / entry * 100, 2) if entry else 0.0
            exit_time = None

    trade = {
        "id": trade_id,
        "time": trade_time,
        "date": now.strftime("%d-%b-%Y"),
        "pattern": pattern_name,
        "signal": signal.signal,
        "option": f"{instrument} {int(atm)} {option_type}",
        "entry_spot": round(spot_price, 2),
        "entry": entry,
        "stop_loss": sl,
        "target": target,
        "rr": round(rr, 2),
        "status": status,
        "exit_price": round(float(exit_price), 2) if exit_price is not None else None,
        "exit_time": exit_time,
        "pnl": round(pnl, 2),
        "pnl_pct": round(pnl_pct, 2),
        "source": "SIM" if simulated else source,
        "confidence": getattr(signal, "confidence", "MEDIUM"),
    }
    st.session_state[f"paper_trades_{instrument}"].append(trade)
    return trade_id


def update_paper_trades(current_spot: float, instrument: str = "NIFTY"):
    """Check open trades and mark them complete if SL or target is hit."""
    init_paper_trades(instrument)
    now = datetime.now(IST)
    for trade in st.session_state[f"paper_trades_{instrument}"]:
        if trade["status"] != "OPEN":
            continue
        entry = trade["entry"]
        sl = trade["stop_loss"]
        target = trade["target"]
        if trade["signal"] == "BUY":
            if current_spot <= sl:
                trade.update(status="LOSS", exit_price=round(sl, 2),
                             exit_time=now.strftime("%H:%M:%S"),
                             pnl=round(sl - entry, 2),
                             pnl_pct=round((sl - entry) / entry * 100, 2))
            elif current_spot >= target:
                trade.update(status="PROFIT", exit_price=round(target, 2),
                             exit_time=now.strftime("%H:%M:%S"),
                             pnl=round(target - entry, 2),
                             pnl_pct=round((target - entry) / entry * 100, 2))
        else:
            if current_spot >= sl:
                trade.update(status="LOSS", exit_price=round(sl, 2),
                             exit_time=now.strftime("%H:%M:%S"),
                             pnl=round(entry - sl, 2),
                             pnl_pct=round((entry - sl) / entry * 100, 2))
            elif current_spot <= target:
                trade.update(status="PROFIT", exit_price=round(target, 2),
                             exit_time=now.strftime("%H:%M:%S"),
                             pnl=round(entry - target, 2),
                             pnl_pct=round((entry - target) / entry * 100, 2))


def get_trades_df(instrument: str = "NIFTY") -> pd.DataFrame:
    init_paper_trades(instrument)
    if not st.session_state[f"paper_trades_{instrument}"]:
        return pd.DataFrame()
    return pd.DataFrame(st.session_state[f"paper_trades_{instrument}"])


def get_paper_trade_summary(instrument: str = "NIFTY") -> Dict:
    df = get_trades_df(instrument)
    if df.empty:
        return {"total": 0, "open": 0, "profit": 0, "loss": 0,
                "total_pnl": 0.0, "win_rate": 0.0}
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


def should_add_new_trade(pattern_name: str, signal_type: str,
                         simulated: bool = False,
                         instrument: str = "NIFTY") -> bool:
    """Avoid duplicate trades for same pattern."""
    init_paper_trades(instrument)
    source = "SIM" if simulated else "AUTO"
    recent = [
        t for t in st.session_state[f"paper_trades_{instrument}"]
        if t["pattern"] == pattern_name
        and t["signal"] == signal_type
        and t["source"] == source
    ]
    return len(recent) == 0


def clear_all_trades(instrument: str = "NIFTY"):
    st.session_state[f"paper_trades_{instrument}"] = []
    st.session_state[f"paper_trade_counter_{instrument}"] = 0
    st.session_state[f"chart_rec_trades_{instrument}"] = []


# ── Chart recommendation trades (spot-level, separate from option paper trades) ──

def add_chart_rec_trade(pattern_name: str, signal: str, entry: float,
                        sl: float, target: float, rr,
                        entry_time: str, exit_time: str, status: str,
                        instrument: str = "NIFTY"):
    """Record a spot-level recommendation trade (no option pricing)."""
    init_paper_trades(instrument)
    key = f"chart_rec_trades_{instrument}"
    trade_key = f"{pattern_name}_{signal}_{entry}_{instrument}"
    if any(t["key"] == trade_key for t in st.session_state[key]):
        return
    st.session_state[key].append({
        "key": trade_key,
        "time": entry_time or "",
        "pattern": pattern_name,
        "signal": signal,
        "entry": round(float(entry), 2),
        "stop_loss": round(float(sl), 2),
        "target": round(float(target), 2),
        "rr": rr,
        "exit_time": exit_time or "",
        "status": status,
    })


def get_chart_rec_trades_df(instrument: str = "NIFTY") -> pd.DataFrame:
    """Return the spot-level chart recommendation trades."""
    init_paper_trades(instrument)
    data = st.session_state.get(f"chart_rec_trades_{instrument}", [])
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if "key" in df.columns:
        df = df.drop(columns=["key"])
    return df
