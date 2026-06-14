"""
Chart Pattern Detection with Professional Scalping Techniques.

Combines classic candlestick patterns with:
  - VWAP bounce / rejection  (John Carter "Mastering the Trade")
  - Opening Range Breakout   (ORB – first 15-min high/low)
  - EMA 9/21 cross with VWAP filter
  - Momentum breakout        (Al Brooks strong-close bar)
  - ATR-based dynamic stop-losses (never fixed ±5)
  - Multi-factor confidence scoring  (VWAP, EMA trend, RSI, volume, time)
  - Time-of-day filter       (penalise first 30 min / last 30 min)
  - Duplicate suppression    (no same-direction signal within 3 bars)
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List
import logging

logger = logging.getLogger(__name__)


@dataclass
class PatternSignal:
    pattern: str
    signal: str        # "BUY" or "SELL"
    index: int
    timestamp: object
    entry: float
    stop_loss: float
    target: float
    risk_reward: float
    confidence: float  # 0-1
    description: str
    color: str = "green"
    counter_trend: bool = False  # True if signal opposes both VWAP & EMA trend
    above_vwap: bool = True      # True if price is on the VWAP side that favours the signal

    def __post_init__(self):
        self.color = "green" if self.signal == "BUY" else "red"
        risk = abs(self.entry - self.stop_loss)
        reward = abs(self.target - self.entry)
        if risk > 0:
            self.risk_reward = round(reward / risk, 2)


# ─── Technical Indicator Helpers ─────────────────────────────────────────────

def _compute_ema(arr: np.ndarray, period: int) -> np.ndarray:
    ema = np.full(len(arr), np.nan)
    if len(arr) < period:
        return ema
    alpha = 2.0 / (period + 1)
    ema[period - 1] = float(np.mean(arr[:period]))
    for i in range(period, len(arr)):
        ema[i] = alpha * arr[i] + (1.0 - alpha) * ema[i - 1]
    return ema


def _compute_atr(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    c = df["close"].values.astype(float)
    n = len(df)
    tr = np.zeros(n)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    atr = np.zeros(n)
    seed = min(period, n)
    atr[:seed] = np.mean(tr[:seed]) if seed > 0 else 1.0
    for i in range(seed, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def _compute_rsi(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    closes = df["close"].values.astype(float)
    n = len(closes)
    rsi = np.full(n, np.nan)
    if n < period + 2:
        return rsi
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))
    for i in range(period, n - 1):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsi[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i + 1] = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _compute_vwap(df: pd.DataFrame) -> np.ndarray:
    """Cumulative VWAP for the session. Returns NaN array if no volume."""
    if "volume" not in df.columns:
        return np.full(len(df), np.nan)
    vol = df["volume"].values.astype(float)
    vol[vol == 0] = np.nan
    tp = ((df["high"] + df["low"] + df["close"]) / 3).values
    cumvol = np.nancumsum(vol)
    cumtpvol = np.nancumsum(tp * vol)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cumvol > 0, cumtpvol / cumvol, np.nan)


def _opening_range(df: pd.DataFrame, minutes: int = 15):
    """Return (or_high, or_low, or_end_idx) for the first N minutes."""
    if "timestamp" not in df.columns or df.empty:
        return np.nan, np.nan, -1
    try:
        ts = pd.to_datetime(df["timestamp"])
        t0 = ts.iloc[0]
        # Anchor to 9:15 IST regardless of first bar
        market_open = t0.normalize().replace(hour=9, minute=15, second=0, microsecond=0)
        if t0.tzinfo is not None:
            import pytz
            market_open = market_open.tz_localize(t0.tzinfo) if market_open.tzinfo is None else market_open
        cutoff = market_open + pd.Timedelta(minutes=minutes)
        mask = ts <= cutoff
        or_df = df[mask]
        if or_df.empty:
            return np.nan, np.nan, -1
        end_idx = int(mask.values.nonzero()[0][-1])
        return float(or_df["high"].max()), float(or_df["low"].min()), end_idx
    except Exception:
        return np.nan, np.nan, -1


def _volume_ratio(df: pd.DataFrame, i: int, lookback: int = 20) -> float:
    if "volume" not in df.columns or i < 1:
        return 1.0
    vol = df["volume"].values.astype(float)
    avg = float(np.mean(vol[max(0, i - lookback):i]))
    return float(vol[i] / avg) if avg > 0 else 1.0


def _time_score(timestamp) -> float:
    """Confidence delta based on time of day."""
    try:
        ts = pd.Timestamp(timestamp)
        m = ts.hour * 60 + ts.minute
        if m < 9 * 60 + 45:     # first 30 min — choppy
            return -0.06
        if m > 15 * 60:          # last 30 min — high slippage
            return -0.10
        if 9 * 60 + 45 <= m <= 10 * 60 + 30:  # prime scalping window
            return 0.05
        if 11 * 60 <= m <= 13 * 60:             # lunch chop
            return -0.03
        return 0.0
    except Exception:
        return 0.0


# ─── Single-Candle Helpers ────────────────────────────────────────────────────

def _body(row) -> float:
    return abs(row["close"] - row["open"])

def _range(row) -> float:
    return row["high"] - row["low"]

def _upper_wick(row) -> float:
    return row["high"] - max(row["open"], row["close"])

def _lower_wick(row) -> float:
    return min(row["open"], row["close"]) - row["low"]


# ─── Classic Candlestick Patterns ────────────────────────────────────────────

def detect_hammer(df: pd.DataFrame) -> List[PatternSignal]:
    signals = []
    for i in range(5, len(df)):
        row = df.iloc[i]
        body = _body(row); total = _range(row)
        lower = _lower_wick(row); upper = _upper_wick(row)
        if total < 1 or body == 0:
            continue
        prior = df["close"].iloc[i - 5:i]
        if prior.iloc[-1] >= prior.iloc[0]:
            continue
        if lower >= 2 * body and upper <= 0.3 * body + 0.1 and body <= 0.35 * total:
            entry = round(row["high"] + 0.5, 2)
            sl = round(row["low"] - 1.0, 2)
            target = round(entry + 2.0 * (entry - sl), 2)
            signals.append(PatternSignal(
                pattern="Hammer", signal="BUY", index=i,
                timestamp=df["timestamp"].iloc[i], entry=entry,
                stop_loss=sl, target=target, risk_reward=0,
                confidence=0.70, description="Hammer – Bullish reversal after downtrend",
            ))
    return signals


def detect_shooting_star(df: pd.DataFrame) -> List[PatternSignal]:
    signals = []
    for i in range(5, len(df)):
        row = df.iloc[i]
        body = _body(row); total = _range(row)
        upper = _upper_wick(row); lower = _lower_wick(row)
        if total < 1 or body == 0:
            continue
        prior = df["close"].iloc[i - 5:i]
        if prior.iloc[-1] <= prior.iloc[0]:
            continue
        if upper >= 2 * body and lower <= 0.3 * body + 0.1 and body <= 0.35 * total:
            entry = round(row["low"] - 0.5, 2)
            sl = round(row["high"] + 1.0, 2)
            target = round(entry - 2.0 * (sl - entry), 2)
            signals.append(PatternSignal(
                pattern="Shooting Star", signal="SELL", index=i,
                timestamp=df["timestamp"].iloc[i], entry=entry,
                stop_loss=sl, target=target, risk_reward=0,
                confidence=0.68, description="Shooting Star – Bearish reversal after uptrend",
            ))
    return signals


def detect_inverted_hammer(df: pd.DataFrame) -> List[PatternSignal]:
    signals = []
    for i in range(5, len(df)):
        row = df.iloc[i]
        body = _body(row); total = _range(row)
        upper = _upper_wick(row); lower = _lower_wick(row)
        if total < 1 or body == 0:
            continue
        prior = df["close"].iloc[i - 5:i]
        if prior.iloc[-1] >= prior.iloc[0]:
            continue
        if upper >= 2 * body and lower <= 0.2 * total and body <= 0.35 * total:
            entry = round(row["high"] + 0.5, 2)
            sl = round(row["low"] - 1.0, 2)
            target = round(entry + 1.5 * (entry - sl), 2)
            signals.append(PatternSignal(
                pattern="Inverted Hammer", signal="BUY", index=i,
                timestamp=df["timestamp"].iloc[i], entry=entry,
                stop_loss=sl, target=target, risk_reward=0,
                confidence=0.60, description="Inverted Hammer – Potential bullish reversal",
            ))
    return signals


def detect_hanging_man(df: pd.DataFrame) -> List[PatternSignal]:
    signals = []
    for i in range(5, len(df)):
        row = df.iloc[i]
        body = _body(row); total = _range(row)
        lower = _lower_wick(row); upper = _upper_wick(row)
        if total < 1 or body == 0:
            continue
        prior = df["close"].iloc[i - 5:i]
        if prior.iloc[-1] <= prior.iloc[0]:
            continue
        if lower >= 2 * body and upper <= 0.3 * body + 0.1 and body <= 0.35 * total:
            entry = round(row["low"] - 0.5, 2)
            sl = round(row["high"] + 1.0, 2)
            target = round(entry - 1.5 * (sl - entry), 2)
            signals.append(PatternSignal(
                pattern="Hanging Man", signal="SELL", index=i,
                timestamp=df["timestamp"].iloc[i], entry=entry,
                stop_loss=sl, target=target, risk_reward=0,
                confidence=0.63, description="Hanging Man – Bearish reversal after uptrend",
            ))
    return signals


def detect_doji(df: pd.DataFrame) -> List[PatternSignal]:
    signals = []
    for i in range(3, len(df)):
        row = df.iloc[i]
        body = _body(row); total = _range(row)
        if total < 1 or body / total >= 0.05:
            continue
        prior = df["close"].iloc[i - 3:i]
        in_uptrend = prior.iloc[-1] > prior.iloc[0]
        if in_uptrend:
            entry = round(row["low"] - 0.5, 2)
            sl = round(row["high"] + 1.0, 2)
            target = round(entry - 1.5 * (sl - entry), 2)
            signals.append(PatternSignal(
                pattern="Doji", signal="SELL", index=i,
                timestamp=df["timestamp"].iloc[i], entry=entry,
                stop_loss=sl, target=target, risk_reward=0,
                confidence=0.53, description="Doji – Indecision at top, potential reversal",
            ))
        else:
            entry = round(row["high"] + 0.5, 2)
            sl = round(row["low"] - 1.0, 2)
            target = round(entry + 1.5 * (entry - sl), 2)
            signals.append(PatternSignal(
                pattern="Doji", signal="BUY", index=i,
                timestamp=df["timestamp"].iloc[i], entry=entry,
                stop_loss=sl, target=target, risk_reward=0,
                confidence=0.53, description="Doji – Indecision at bottom, potential reversal",
            ))
    return signals


def detect_dragonfly_doji(df: pd.DataFrame) -> List[PatternSignal]:
    signals = []
    for i in range(3, len(df)):
        row = df.iloc[i]
        body = _body(row); total = _range(row)
        if total < 1:
            continue
        if body / total < 0.07 and _upper_wick(row) / total < 0.05 and _lower_wick(row) / total > 0.7:
            entry = round(row["close"] + 0.5, 2)
            sl = round(row["low"] - 1.0, 2)
            target = round(entry + 2.0 * (entry - sl), 2)
            signals.append(PatternSignal(
                pattern="Dragonfly Doji", signal="BUY", index=i,
                timestamp=df["timestamp"].iloc[i], entry=entry,
                stop_loss=sl, target=target, risk_reward=0,
                confidence=0.66, description="Dragonfly Doji – Strong bullish reversal",
            ))
    return signals


def detect_gravestone_doji(df: pd.DataFrame) -> List[PatternSignal]:
    signals = []
    for i in range(3, len(df)):
        row = df.iloc[i]
        body = _body(row); total = _range(row)
        if total < 1:
            continue
        if body / total < 0.07 and _lower_wick(row) / total < 0.05 and _upper_wick(row) / total > 0.7:
            entry = round(row["close"] - 0.5, 2)
            sl = round(row["high"] + 1.0, 2)
            target = round(entry - 2.0 * (sl - entry), 2)
            signals.append(PatternSignal(
                pattern="Gravestone Doji", signal="SELL", index=i,
                timestamp=df["timestamp"].iloc[i], entry=entry,
                stop_loss=sl, target=target, risk_reward=0,
                confidence=0.66, description="Gravestone Doji – Strong bearish reversal",
            ))
    return signals


def detect_pin_bar(df: pd.DataFrame) -> List[PatternSignal]:
    signals = []
    for i in range(5, len(df)):
        row = df.iloc[i]
        body = _body(row); total = _range(row)
        upper = _upper_wick(row); lower = _lower_wick(row)
        if total < 2:
            continue
        prior = df["close"].iloc[i - 5:i]
        in_uptrend = prior.iloc[-1] > prior.iloc[0]
        if in_uptrend and upper >= 0.65 * total and body <= 0.25 * total and lower <= 0.2 * total:
            entry = round(row["low"] - 0.5, 2)
            sl = round(row["high"] + 1.0, 2)
            target = round(entry - 2.5 * (sl - entry), 2)
            signals.append(PatternSignal(
                pattern="Pin Bar (Bearish)", signal="SELL", index=i,
                timestamp=df["timestamp"].iloc[i], entry=entry,
                stop_loss=sl, target=target, risk_reward=0,
                confidence=0.71, description="Bearish Pin Bar – Rejection of highs",
            ))
        elif not in_uptrend and lower >= 0.65 * total and body <= 0.25 * total and upper <= 0.2 * total:
            entry = round(row["high"] + 0.5, 2)
            sl = round(row["low"] - 1.0, 2)
            target = round(entry + 2.5 * (entry - sl), 2)
            signals.append(PatternSignal(
                pattern="Pin Bar (Bullish)", signal="BUY", index=i,
                timestamp=df["timestamp"].iloc[i], entry=entry,
                stop_loss=sl, target=target, risk_reward=0,
                confidence=0.71, description="Bullish Pin Bar – Rejection of lows",
            ))
    return signals


def detect_engulfing(df: pd.DataFrame) -> List[PatternSignal]:
    signals = []
    for i in range(6, len(df)):
        curr = df.iloc[i]; prev = df.iloc[i - 1]
        curr_body = _body(curr); prev_body = _body(prev)
        if prev_body < 1:
            continue
        prior = df["close"].iloc[i - 6:i - 1]
        in_down = prior.iloc[-1] < prior.iloc[0]
        in_up   = prior.iloc[-1] > prior.iloc[0]
        if (in_down and prev["close"] < prev["open"] and curr["close"] > curr["open"]
                and curr["open"] < prev["close"] and curr["close"] > prev["open"]
                and curr_body > prev_body):
            entry = round(curr["close"] + 0.5, 2)
            sl    = round(min(curr["low"], prev["low"]) - 1.0, 2)
            target = round(entry + 2.0 * (entry - sl), 2)
            signals.append(PatternSignal(
                pattern="Bullish Engulfing", signal="BUY", index=i,
                timestamp=df["timestamp"].iloc[i], entry=entry,
                stop_loss=sl, target=target, risk_reward=0,
                confidence=0.76, description="Bullish Engulfing – Strong reversal",
            ))
        elif (in_up and prev["close"] > prev["open"] and curr["close"] < curr["open"]
              and curr["open"] > prev["close"] and curr["close"] < prev["open"]
              and curr_body > prev_body):
            entry = round(curr["close"] - 0.5, 2)
            sl    = round(max(curr["high"], prev["high"]) + 1.0, 2)
            target = round(entry - 2.0 * (sl - entry), 2)
            signals.append(PatternSignal(
                pattern="Bearish Engulfing", signal="SELL", index=i,
                timestamp=df["timestamp"].iloc[i], entry=entry,
                stop_loss=sl, target=target, risk_reward=0,
                confidence=0.76, description="Bearish Engulfing – Strong reversal",
            ))
    return signals


def detect_inside_bar(df: pd.DataFrame) -> List[PatternSignal]:
    signals = []
    for i in range(5, len(df)):
        curr = df.iloc[i]; prev = df.iloc[i - 1]
        if curr["high"] < prev["high"] and curr["low"] > prev["low"] and _range(prev) > 2:
            prior = df["close"].iloc[i - 5:i - 1]
            in_up = prior.iloc[-1] > prior.iloc[0]
            if in_up:
                entry = round(prev["high"] + 0.5, 2)
                sl    = round(prev["low"] - 1.0, 2)
                target = round(entry + 2.0 * (entry - sl), 2)
                signals.append(PatternSignal(
                    pattern="Inside Bar (Bullish)", signal="BUY", index=i,
                    timestamp=df["timestamp"].iloc[i], entry=entry,
                    stop_loss=sl, target=target, risk_reward=0,
                    confidence=0.63, description="Inside Bar – Bullish breakout pending",
                ))
            else:
                entry = round(prev["low"] - 0.5, 2)
                sl    = round(prev["high"] + 1.0, 2)
                target = round(entry - 2.0 * (sl - entry), 2)
                signals.append(PatternSignal(
                    pattern="Inside Bar (Bearish)", signal="SELL", index=i,
                    timestamp=df["timestamp"].iloc[i], entry=entry,
                    stop_loss=sl, target=target, risk_reward=0,
                    confidence=0.63, description="Inside Bar – Bearish breakout pending",
                ))
    return signals


def detect_morning_star(df: pd.DataFrame) -> List[PatternSignal]:
    signals = []
    for i in range(7, len(df)):
        c1 = df.iloc[i - 2]; c2 = df.iloc[i - 1]; c3 = df.iloc[i]
        prior = df["close"].iloc[i - 7:i - 2]
        if prior.iloc[-1] >= prior.iloc[0]:
            continue
        if (c1["close"] < c1["open"] and c3["close"] > c3["open"]
                and _body(c2) < 0.4 * _body(c1) and _body(c3) > 0.5 * _body(c1)
                and c3["close"] > c1["open"] + (c1["close"] - c1["open"]) * 0.3):
            entry = round(c3["close"] + 0.5, 2)
            sl    = round(c2["low"] - 1.0, 2)
            target = round(entry + 2.5 * (entry - sl), 2)
            signals.append(PatternSignal(
                pattern="Morning Star", signal="BUY", index=i,
                timestamp=df["timestamp"].iloc[i], entry=entry,
                stop_loss=sl, target=target, risk_reward=0,
                confidence=0.78, description="Morning Star – Strong bullish reversal",
            ))
    return signals


def detect_evening_star(df: pd.DataFrame) -> List[PatternSignal]:
    signals = []
    for i in range(7, len(df)):
        c1 = df.iloc[i - 2]; c2 = df.iloc[i - 1]; c3 = df.iloc[i]
        prior = df["close"].iloc[i - 7:i - 2]
        if prior.iloc[-1] <= prior.iloc[0]:
            continue
        if (c1["close"] > c1["open"] and c3["close"] < c3["open"]
                and _body(c2) < 0.4 * _body(c1) and _body(c3) > 0.5 * _body(c1)
                and c3["close"] < c1["open"] + (c1["close"] - c1["open"]) * 0.3):
            entry = round(c3["close"] - 0.5, 2)
            sl    = round(c2["high"] + 1.0, 2)
            target = round(entry - 2.5 * (sl - entry), 2)
            signals.append(PatternSignal(
                pattern="Evening Star", signal="SELL", index=i,
                timestamp=df["timestamp"].iloc[i], entry=entry,
                stop_loss=sl, target=target, risk_reward=0,
                confidence=0.78, description="Evening Star – Strong bearish reversal",
            ))
    return signals


# ─── Multi-Candle Chart Patterns ──────────────────────────────────────────────

def _find_peaks(arr: np.ndarray, min_dist: int = 3) -> List[int]:
    peaks = []
    for i in range(1, len(arr) - 1):
        if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
            if not peaks or (i - peaks[-1]) >= min_dist:
                peaks.append(i)
            elif arr[i] > arr[peaks[-1]]:
                peaks[-1] = i
    return peaks


def _find_troughs(arr: np.ndarray, min_dist: int = 3) -> List[int]:
    troughs = []
    for i in range(1, len(arr) - 1):
        if arr[i] < arr[i - 1] and arr[i] < arr[i + 1]:
            if not troughs or (i - troughs[-1]) >= min_dist:
                troughs.append(i)
            elif arr[i] < arr[troughs[-1]]:
                troughs[-1] = i
    return troughs


def detect_double_top(df: pd.DataFrame, window: int = 30, tol: float = 0.003) -> List[PatternSignal]:
    signals = []
    if len(df) < window:
        return signals
    for i in range(window, len(df)):
        seg = df.iloc[i - window:i]
        highs = seg["high"].values; lows = seg["low"].values
        peaks = _find_peaks(highs, min_dist=5)
        if len(peaks) < 2:
            continue
        p1, p2 = peaks[-2], peaks[-1]
        if abs(highs[p1] - highs[p2]) / highs[p1] > tol:
            continue
        valley = lows[p1:p2]
        if len(valley) == 0:
            continue
        neck = float(np.min(valley))
        if df["close"].iloc[i - 1] < neck:
            top = max(highs[p1], highs[p2])
            entry = round(neck - 0.5, 2)
            sl    = round(top + 1.0, 2)
            target = round(neck - (top - neck), 2)
            signals.append(PatternSignal(
                pattern="Double Top", signal="SELL", index=i - 1,
                timestamp=df["timestamp"].iloc[i - 1], entry=entry,
                stop_loss=sl, target=target, risk_reward=0,
                confidence=0.73, description=f"Double Top at {top:.0f} – Breakdown below neck {neck:.0f}",
            ))
    return signals


def detect_double_bottom(df: pd.DataFrame, window: int = 30, tol: float = 0.003) -> List[PatternSignal]:
    signals = []
    if len(df) < window:
        return signals
    for i in range(window, len(df)):
        seg = df.iloc[i - window:i]
        lows = seg["low"].values; highs = seg["high"].values
        troughs = _find_troughs(lows, min_dist=5)
        if len(troughs) < 2:
            continue
        t1, t2 = troughs[-2], troughs[-1]
        if abs(lows[t1] - lows[t2]) / lows[t1] > tol:
            continue
        peak_highs = highs[t1:t2]
        if len(peak_highs) == 0:
            continue
        neck = float(np.max(peak_highs))
        if df["close"].iloc[i - 1] > neck:
            bot = min(lows[t1], lows[t2])
            entry = round(neck + 0.5, 2)
            sl    = round(bot - 1.0, 2)
            target = round(neck + (neck - bot), 2)
            signals.append(PatternSignal(
                pattern="Double Bottom", signal="BUY", index=i - 1,
                timestamp=df["timestamp"].iloc[i - 1], entry=entry,
                stop_loss=sl, target=target, risk_reward=0,
                confidence=0.73, description=f"Double Bottom at {bot:.0f} – Breakout above neck {neck:.0f}",
            ))
    return signals


def detect_head_and_shoulders(df: pd.DataFrame, window: int = 40) -> List[PatternSignal]:
    signals = []
    if len(df) < window:
        return signals
    for i in range(window, len(df)):
        seg = df.iloc[i - window:i]
        highs = seg["high"].values; lows = seg["low"].values
        peaks = _find_peaks(highs, min_dist=4)
        if len(peaks) < 3:
            continue
        ls, head, rs = peaks[-3], peaks[-2], peaks[-1]
        if not (highs[head] > highs[ls] and highs[head] > highs[rs]):
            continue
        if abs(highs[ls] - highs[rs]) / highs[ls] > 0.015:
            continue
        troughs = _find_troughs(lows[ls:rs + 1], min_dist=2)
        if len(troughs) < 2:
            continue
        neck = float(np.mean([lows[ls + troughs[0]], lows[ls + troughs[-1]]]))
        if df["close"].iloc[i - 1] < neck:
            entry = round(neck - 0.5, 2)
            sl    = round(highs[rs] + 1.0, 2)
            target = round(neck - (highs[head] - neck), 2)
            signals.append(PatternSignal(
                pattern="Head & Shoulders", signal="SELL", index=i - 1,
                timestamp=df["timestamp"].iloc[i - 1], entry=entry,
                stop_loss=sl, target=target, risk_reward=0,
                confidence=0.80, description=f"H&S – Breakdown below neckline {neck:.0f}",
            ))
    return signals


def detect_inverse_head_and_shoulders(df: pd.DataFrame, window: int = 40) -> List[PatternSignal]:
    signals = []
    if len(df) < window:
        return signals
    for i in range(window, len(df)):
        seg = df.iloc[i - window:i]
        highs = seg["high"].values; lows = seg["low"].values
        troughs = _find_troughs(lows, min_dist=4)
        if len(troughs) < 3:
            continue
        ls, head, rs = troughs[-3], troughs[-2], troughs[-1]
        if not (lows[head] < lows[ls] and lows[head] < lows[rs]):
            continue
        if abs(lows[ls] - lows[rs]) / lows[ls] > 0.015:
            continue
        peaks = _find_peaks(highs[ls:rs + 1], min_dist=2)
        if len(peaks) < 2:
            continue
        neck = float(np.mean([highs[ls + peaks[0]], highs[ls + peaks[-1]]]))
        if df["close"].iloc[i - 1] > neck:
            entry = round(neck + 0.5, 2)
            sl    = round(lows[rs] - 1.0, 2)
            target = round(neck + (neck - lows[head]), 2)
            signals.append(PatternSignal(
                pattern="Inv. Head & Shoulders", signal="BUY", index=i - 1,
                timestamp=df["timestamp"].iloc[i - 1], entry=entry,
                stop_loss=sl, target=target, risk_reward=0,
                confidence=0.80, description=f"Inv. H&S – Breakout above neckline {neck:.0f}",
            ))
    return signals


def detect_ascending_triangle(df: pd.DataFrame, window: int = 30) -> List[PatternSignal]:
    signals = []
    if len(df) < window:
        return signals
    for i in range(window, len(df)):
        seg = df.iloc[i - window:i]
        highs = seg["high"].values; lows = seg["low"].values
        recent_highs = highs[-10:]
        resistance = float(np.mean(recent_highs))
        if np.std(recent_highs) / resistance >= 0.005:
            continue
        slope = float(np.polyfit(range(len(lows)), lows, 1)[0])
        if slope <= 0:
            continue
        if df["close"].iloc[i - 1] > resistance:
            entry = round(resistance + 0.5, 2)
            sl    = round(float(lows[-1]) - 1.0, 2)
            target = round(resistance + (resistance - float(np.min(lows[-window:]))), 2)
            signals.append(PatternSignal(
                pattern="Ascending Triangle", signal="BUY", index=i - 1,
                timestamp=df["timestamp"].iloc[i - 1], entry=entry,
                stop_loss=sl, target=target, risk_reward=0,
                confidence=0.71, description=f"Ascending Triangle breakout above {resistance:.0f}",
            ))
    return signals


def detect_descending_triangle(df: pd.DataFrame, window: int = 30) -> List[PatternSignal]:
    signals = []
    if len(df) < window:
        return signals
    for i in range(window, len(df)):
        seg = df.iloc[i - window:i]
        highs = seg["high"].values; lows = seg["low"].values
        recent_lows = lows[-10:]
        support = float(np.mean(recent_lows))
        if np.std(recent_lows) / support >= 0.005:
            continue
        slope = float(np.polyfit(range(len(highs)), highs, 1)[0])
        if slope >= 0:
            continue
        if df["close"].iloc[i - 1] < support:
            entry = round(support - 0.5, 2)
            sl    = round(float(highs[-1]) + 1.0, 2)
            target = round(support - (float(np.max(highs[-window:])) - support), 2)
            signals.append(PatternSignal(
                pattern="Descending Triangle", signal="SELL", index=i - 1,
                timestamp=df["timestamp"].iloc[i - 1], entry=entry,
                stop_loss=sl, target=target, risk_reward=0,
                confidence=0.71, description=f"Descending Triangle breakdown below {support:.0f}",
            ))
    return signals


def detect_bull_flag(df: pd.DataFrame, pole_bars: int = 10, flag_bars: int = 10) -> List[PatternSignal]:
    signals = []
    if len(df) < pole_bars + flag_bars + 5:
        return signals
    for i in range(pole_bars + flag_bars, len(df)):
        pole = df.iloc[i - pole_bars - flag_bars:i - flag_bars]
        flag = df.iloc[i - flag_bars:i]
        pole_gain = (pole["close"].iloc[-1] - pole["close"].iloc[0]) / pole["close"].iloc[0]
        if pole_gain < 0.01:
            continue
        flag_range = flag["high"].max() - flag["low"].min()
        pole_range = pole["high"].max() - pole["low"].min()
        if flag_range > 0.5 * pole_range:
            continue
        slope = float(np.polyfit(range(len(flag)), flag["close"].values, 1)[0])
        if slope > 0.2:
            continue
        if df["close"].iloc[i - 1] > flag["high"].max():
            entry = round(float(flag["high"].max()) + 0.5, 2)
            sl    = round(float(flag["low"].min()) - 1.0, 2)
            target = round(entry + pole_range, 2)
            signals.append(PatternSignal(
                pattern="Bull Flag", signal="BUY", index=i - 1,
                timestamp=df["timestamp"].iloc[i - 1], entry=entry,
                stop_loss=sl, target=target, risk_reward=0,
                confidence=0.74, description="Bull Flag – Bullish continuation breakout",
            ))
    return signals


def detect_bear_flag(df: pd.DataFrame, pole_bars: int = 10, flag_bars: int = 10) -> List[PatternSignal]:
    signals = []
    if len(df) < pole_bars + flag_bars + 5:
        return signals
    for i in range(pole_bars + flag_bars, len(df)):
        pole = df.iloc[i - pole_bars - flag_bars:i - flag_bars]
        flag = df.iloc[i - flag_bars:i]
        pole_loss = (pole["close"].iloc[0] - pole["close"].iloc[-1]) / pole["close"].iloc[0]
        if pole_loss < 0.01:
            continue
        flag_range = flag["high"].max() - flag["low"].min()
        pole_range = pole["high"].max() - pole["low"].min()
        if flag_range > 0.5 * pole_range:
            continue
        slope = float(np.polyfit(range(len(flag)), flag["close"].values, 1)[0])
        if slope < -0.2:
            continue
        if df["close"].iloc[i - 1] < flag["low"].min():
            entry = round(float(flag["low"].min()) - 0.5, 2)
            sl    = round(float(flag["high"].max()) + 1.0, 2)
            target = round(entry - pole_range, 2)
            signals.append(PatternSignal(
                pattern="Bear Flag", signal="SELL", index=i - 1,
                timestamp=df["timestamp"].iloc[i - 1], entry=entry,
                stop_loss=sl, target=target, risk_reward=0,
                confidence=0.74, description="Bear Flag – Bearish continuation breakdown",
            ))
    return signals


# ─── Professional Scalping Patterns ──────────────────────────────────────────

def detect_vwap_bounce(df: pd.DataFrame, vwap: np.ndarray,
                        atr: np.ndarray) -> List[PatternSignal]:
    """
    VWAP Bounce/Rejection – price touches VWAP and reverses.
    High-probability scalp setup (John Carter, Tom Williams).
    """
    signals = []
    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values
    for i in range(5, len(df)):
        if np.isnan(vwap[i]) or np.isnan(atr[i]) or atr[i] <= 0:
            continue
        v = vwap[i]; band = v * 0.0012  # 0.12% touch band
        atr_v = atr[i]

        # Bullish bounce: wick penetrates below VWAP, candle closes back above
        if lows[i] < v - band and closes[i] > v + band:
            prior = closes[max(0, i - 5):i]
            if len(prior) > 1 and prior[-1] < prior[0]:   # down into VWAP
                entry = round(closes[i] + 0.5, 2)
                sl    = round(lows[i] - 0.5 * atr_v, 2)
                target = round(entry + 2.0 * (entry - sl), 2)
                signals.append(PatternSignal(
                    pattern="VWAP Bounce (Bullish)", signal="BUY", index=i,
                    timestamp=df["timestamp"].iloc[i], entry=entry,
                    stop_loss=sl, target=target, risk_reward=0,
                    confidence=0.74, description=f"VWAP bounce at {v:.0f} – buyers stepped in",
                ))

        # Bearish rejection: wick above VWAP, candle closes back below
        elif highs[i] > v + band and closes[i] < v - band:
            prior = closes[max(0, i - 5):i]
            if len(prior) > 1 and prior[-1] > prior[0]:   # up into VWAP
                entry = round(closes[i] - 0.5, 2)
                sl    = round(highs[i] + 0.5 * atr_v, 2)
                target = round(entry - 2.0 * (sl - entry), 2)
                signals.append(PatternSignal(
                    pattern="VWAP Rejection (Bearish)", signal="SELL", index=i,
                    timestamp=df["timestamp"].iloc[i], entry=entry,
                    stop_loss=sl, target=target, risk_reward=0,
                    confidence=0.74, description=f"VWAP rejection at {v:.0f} – sellers defending",
                ))
    return signals


def detect_orb_breakout(df: pd.DataFrame, or_high: float, or_low: float,
                          or_end_idx: int, atr: np.ndarray) -> List[PatternSignal]:
    """
    Opening Range Breakout (ORB) – first confirmed close outside first-15-min range.
    One signal per direction per session.
    """
    signals = []
    if np.isnan(or_high) or np.isnan(or_low) or or_end_idx < 0:
        return signals
    or_range = or_high - or_low
    if or_range <= 0:
        return signals
    closes = df["close"].values
    bullish_done = bearish_done = False
    for i in range(or_end_idx + 1, len(df)):
        if np.isnan(atr[i]):
            continue
        atr_v = atr[i]
        if not bullish_done and closes[i] > or_high:
            # Check no earlier bar already closed above
            if not any(closes[or_end_idx + 1:i] > or_high):
                entry  = round(or_high + 0.5, 2)
                sl     = round(or_high - 0.4 * atr_v, 2)
                target = round(entry + or_range, 2)
                signals.append(PatternSignal(
                    pattern="ORB Breakout (Bullish)", signal="BUY", index=i,
                    timestamp=df["timestamp"].iloc[i], entry=entry,
                    stop_loss=sl, target=target, risk_reward=0,
                    confidence=0.77, description=f"ORB breakout above {or_high:.0f} (range {or_range:.0f})",
                ))
                bullish_done = True
        if not bearish_done and closes[i] < or_low:
            if not any(closes[or_end_idx + 1:i] < or_low):
                entry  = round(or_low - 0.5, 2)
                sl     = round(or_low + 0.4 * atr_v, 2)
                target = round(entry - or_range, 2)
                signals.append(PatternSignal(
                    pattern="ORB Breakdown (Bearish)", signal="SELL", index=i,
                    timestamp=df["timestamp"].iloc[i], entry=entry,
                    stop_loss=sl, target=target, risk_reward=0,
                    confidence=0.77, description=f"ORB breakdown below {or_low:.0f} (range {or_range:.0f})",
                ))
                bearish_done = True
        if bullish_done and bearish_done:
            break
    return signals


def detect_ema_cross(df: pd.DataFrame, ema9: np.ndarray, ema21: np.ndarray,
                      vwap: np.ndarray, atr: np.ndarray) -> List[PatternSignal]:
    """
    EMA 9/21 crossover with VWAP-side filter.
    Bullish: EMA9 > EMA21 and price above VWAP.
    Bearish: EMA9 < EMA21 and price below VWAP.
    """
    signals = []
    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values
    for i in range(22, len(df)):
        if (np.isnan(ema9[i]) or np.isnan(ema21[i]) or np.isnan(atr[i])
                or np.isnan(ema9[i - 1]) or np.isnan(ema21[i - 1])):
            continue
        atr_v = atr[i]
        vwap_ok_buy  = np.isnan(vwap[i]) or closes[i] > vwap[i]
        vwap_ok_sell = np.isnan(vwap[i]) or closes[i] < vwap[i]

        if ema9[i - 1] <= ema21[i - 1] and ema9[i] > ema21[i]:
            entry  = round(closes[i] + 0.5, 2)
            sl     = round(min(lows[i], lows[i - 1]) - 0.4 * atr_v, 2)
            target = round(entry + 2.0 * (entry - sl), 2)
            conf   = 0.72 if vwap_ok_buy else 0.59
            signals.append(PatternSignal(
                pattern="EMA 9/21 Cross (Bullish)", signal="BUY", index=i,
                timestamp=df["timestamp"].iloc[i], entry=entry,
                stop_loss=sl, target=target, risk_reward=0,
                confidence=conf, description=f"EMA9 crossed above EMA21 at {closes[i]:.0f}",
            ))

        elif ema9[i - 1] >= ema21[i - 1] and ema9[i] < ema21[i]:
            entry  = round(closes[i] - 0.5, 2)
            sl     = round(max(highs[i], highs[i - 1]) + 0.4 * atr_v, 2)
            target = round(entry - 2.0 * (sl - entry), 2)
            conf   = 0.72 if vwap_ok_sell else 0.59
            signals.append(PatternSignal(
                pattern="EMA 9/21 Cross (Bearish)", signal="SELL", index=i,
                timestamp=df["timestamp"].iloc[i], entry=entry,
                stop_loss=sl, target=target, risk_reward=0,
                confidence=conf, description=f"EMA9 crossed below EMA21 at {closes[i]:.0f}",
            ))
    return signals


def detect_momentum_breakout(df: pd.DataFrame, vwap: np.ndarray,
                               atr: np.ndarray) -> List[PatternSignal]:
    """
    Momentum breakout – candle closes in top/bottom 25% of range with body > 60% of range.
    Al Brooks "strong close" concept: buyers/sellers completely in control.
    Only fires when the current ATR is above median (active market).
    """
    signals = []
    if len(df) < 20:
        return signals
    closes = df["close"].values
    opens  = df["open"].values
    highs  = df["high"].values
    lows   = df["low"].values
    atr_median = float(np.median(atr[~np.isnan(atr)])) if np.any(~np.isnan(atr)) else 1.0

    for i in range(10, len(df)):
        if np.isnan(atr[i]) or atr[i] < 0.5 * atr_median:
            continue  # Skip low-volatility bars
        total = highs[i] - lows[i]
        if total <= 0:
            continue
        body_pos = (closes[i] - lows[i]) / total  # 0=at low, 1=at high
        body_frac = abs(closes[i] - opens[i]) / total

        if body_frac >= 0.60 and body_pos >= 0.75 and closes[i] > opens[i]:
            prior_trend = closes[i] > float(np.mean(closes[max(0, i - 10):i]))
            vwap_align  = np.isnan(vwap[i]) or closes[i] > vwap[i]
            entry  = round(closes[i] + 0.5, 2)
            sl     = round(lows[i] - 0.5 * atr[i], 2)
            target = round(entry + 1.8 * (entry - sl), 2)
            conf   = 0.67 + (0.05 if prior_trend else 0) + (0.04 if vwap_align else 0)
            signals.append(PatternSignal(
                pattern="Momentum Breakout (Bullish)", signal="BUY", index=i,
                timestamp=df["timestamp"].iloc[i], entry=entry,
                stop_loss=sl, target=target, risk_reward=0,
                confidence=round(conf, 2), description=f"Strong bullish close at {closes[i]:.0f}",
            ))

        elif body_frac >= 0.60 and body_pos <= 0.25 and closes[i] < opens[i]:
            prior_trend = closes[i] < float(np.mean(closes[max(0, i - 10):i]))
            vwap_align  = np.isnan(vwap[i]) or closes[i] < vwap[i]
            entry  = round(closes[i] - 0.5, 2)
            sl     = round(highs[i] + 0.5 * atr[i], 2)
            target = round(entry - 1.8 * (sl - entry), 2)
            conf   = 0.67 + (0.05 if prior_trend else 0) + (0.04 if vwap_align else 0)
            signals.append(PatternSignal(
                pattern="Momentum Breakout (Bearish)", signal="SELL", index=i,
                timestamp=df["timestamp"].iloc[i], entry=entry,
                stop_loss=sl, target=target, risk_reward=0,
                confidence=round(conf, 2), description=f"Strong bearish close at {closes[i]:.0f}",
            ))
    return signals


def detect_rsi_reversal(df: pd.DataFrame, rsi: np.ndarray, atr: np.ndarray,
                          ema9: np.ndarray, ema21: np.ndarray) -> List[PatternSignal]:
    """
    RSI extreme + EMA trend alignment reversal.
    Oversold (<35) with EMA9 starting to turn up → BUY scalp.
    Overbought (>65) with EMA9 starting to turn down → SELL scalp.
    """
    signals = []
    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values
    for i in range(22, len(df)):
        if (np.isnan(rsi[i]) or np.isnan(rsi[i - 1])
                or np.isnan(atr[i]) or atr[i] <= 0
                or np.isnan(ema9[i]) or np.isnan(ema9[i - 1])):
            continue
        atr_v = atr[i]

        # Oversold bounce: RSI was <35, now rising; EMA9 also ticking up
        if rsi[i - 1] < 35 and rsi[i] > rsi[i - 1] and ema9[i] >= ema9[i - 1]:
            entry  = round(closes[i] + 0.5, 2)
            sl     = round(lows[i] - 0.6 * atr_v, 2)
            target = round(entry + 2.0 * (entry - sl), 2)
            signals.append(PatternSignal(
                pattern="RSI Oversold Bounce", signal="BUY", index=i,
                timestamp=df["timestamp"].iloc[i], entry=entry,
                stop_loss=sl, target=target, risk_reward=0,
                confidence=0.69, description=f"RSI oversold bounce ({rsi[i-1]:.0f}→{rsi[i]:.0f})",
            ))

        # Overbought reversal: RSI was >65, now falling; EMA9 ticking down
        elif rsi[i - 1] > 65 and rsi[i] < rsi[i - 1] and ema9[i] <= ema9[i - 1]:
            entry  = round(closes[i] - 0.5, 2)
            sl     = round(highs[i] + 0.6 * atr_v, 2)
            target = round(entry - 2.0 * (sl - entry), 2)
            signals.append(PatternSignal(
                pattern="RSI Overbought Reversal", signal="SELL", index=i,
                timestamp=df["timestamp"].iloc[i], entry=entry,
                stop_loss=sl, target=target, risk_reward=0,
                confidence=0.69, description=f"RSI overbought reversal ({rsi[i-1]:.0f}→{rsi[i]:.0f})",
            ))
    return signals


def detect_range_breakout(df: pd.DataFrame, atr: np.ndarray,
                            vwap: np.ndarray, look: int = 8) -> List[PatternSignal]:
    """
    Range / Consolidation Breakout (Darvas Box, Bollinger squeeze concept).

    Big traders accumulate inside a tight range, then a wide-range expansion
    candle breaks out on volume. We catch THAT breakout candle — the single
    most important bar — instead of chasing the move later near the top.

    Conditions:
      • prior `look` bars form a tight box (range ≤ 2.5×ATR)  → consolidation
      • current candle closes above the box high               → breakout
      • current candle is green with range ≥ 1.1×ATR           → expansion
    """
    signals = []
    if len(df) < look + 3:
        return signals
    opens  = df["open"].values
    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values
    for i in range(look + 2, len(df)):
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue
        atr_v = atr[i]
        box_high = float(highs[i - look:i].max())
        box_low  = float(lows[i - look:i].min())
        box_range = box_high - box_low
        if box_range <= 0 or box_range > 2.5 * atr_v:
            continue   # not a tight consolidation
        cur_range = highs[i] - lows[i]
        # Breakout candle: closes above box on an expansion green bar
        if (closes[i] > box_high and closes[i] > opens[i]
                and cur_range >= 1.1 * atr_v):
            entry  = round(closes[i] + 0.5, 2)
            # Structural stop just back inside the box / below breakout bar
            sl     = round(min(lows[i], box_high) - 0.3 * atr_v, 2)
            risk   = entry - sl
            target = round(entry + 1.6 * risk, 2)
            signals.append(PatternSignal(
                pattern="Range Breakout (Bullish)", signal="BUY", index=i,
                timestamp=df["timestamp"].iloc[i], entry=entry,
                stop_loss=sl, target=target, risk_reward=0,
                confidence=0.80, description=f"Box breakout above {box_high:.0f} on expansion candle",
            ))
    return signals


def detect_ema_pullback(df: pd.DataFrame, ema9: np.ndarray, ema21: np.ndarray,
                          vwap: np.ndarray, atr: np.ndarray) -> List[PatternSignal]:
    """
    EMA Pullback Continuation – Al Brooks "High-2" / trend-pullback entry.

    The highest-probability trend entry is NOT the breakout but the first
    pullback to the rising EMA inside an established uptrend. We buy the
    resumption bar (price dipped to EMA9, then closed back up making a higher
    bar) so we ride the trend without chasing the top.

    Conditions:
      • EMA9 > EMA21 and price above VWAP        → confirmed uptrend
      • previous bar's low tagged the EMA9 zone  → healthy pullback
      • current bar closes green above prev high → resumption
    """
    signals = []
    opens  = df["open"].values
    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values
    for i in range(23, len(df)):
        if (np.isnan(ema9[i]) or np.isnan(ema21[i]) or np.isnan(atr[i])
                or atr[i] <= 0 or np.isnan(ema9[i - 1])):
            continue
        atr_v = atr[i]
        # Established uptrend with VWAP support
        uptrend = ema9[i] > ema21[i] and (np.isnan(vwap[i]) or closes[i] > vwap[i])
        if not uptrend:
            continue
        # Previous bar pulled back to the EMA9 zone (low near EMA9)
        pulled_back = (lows[i - 1] <= ema9[i - 1] + 0.4 * atr_v
                       and lows[i - 1] >= ema21[i - 1] - 0.6 * atr_v)
        # Current bar resumes: green and closes above previous high (higher bar)
        resumed = closes[i] > opens[i] and closes[i] > highs[i - 1]
        if pulled_back and resumed:
            entry  = round(closes[i] + 0.5, 2)
            sl     = round(min(lows[i], lows[i - 1]) - 0.3 * atr_v, 2)
            risk   = entry - sl
            target = round(entry + 1.5 * risk, 2)
            signals.append(PatternSignal(
                pattern="EMA Pullback (Bullish)", signal="BUY", index=i,
                timestamp=df["timestamp"].iloc[i], entry=entry,
                stop_loss=sl, target=target, risk_reward=0,
                confidence=0.79, description=f"Trend pullback to EMA9 then resumption at {closes[i]:.0f}",
            ))
    return signals


# ─── Multi-Factor Post-Processor ─────────────────────────────────────────────

def _apply_multi_factor(signals: List[PatternSignal], df: pd.DataFrame,
                         vwap: np.ndarray, atr: np.ndarray,
                         ema9: np.ndarray, ema21: np.ndarray,
                         rsi: np.ndarray) -> List[PatternSignal]:
    """
    For every signal:
    1. Widen stop to ATR-minimum if it was set too tight (common with fixed offsets).
    2. Score VWAP alignment, EMA trend, RSI zone, volume, time-of-day.
    3. Recompute risk_reward from updated levels.
    """
    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values
    enhanced = []

    for sig in signals:
        i = sig.index
        if i >= len(df):
            enhanced.append(sig); continue

        # ── ATR-based risk normalisation ───────────────────────────────────
        # Scalping wants a STOP wide enough to survive normal noise, paired
        # with a CLOSE, achievable target (so the hit-rate stays high). We
        # clamp every signal's risk into a sensible ATR band and set the
        # target at 1.4× that risk (R:R ≈ 1:1.4).
        if not np.isnan(atr[i]) and atr[i] > 0:
            atr_v = atr[i]
            current_risk = abs(sig.entry - sig.stop_loss)
            # Stop floor 1.0×ATR, ceiling 1.8×ATR → "thoda jyada" SL
            risk = min(max(current_risk, 1.0 * atr_v), 1.8 * atr_v)
            if sig.signal == "BUY":
                sig.stop_loss = round(sig.entry - risk, 2)
                sig.target    = round(sig.entry + 1.4 * risk, 2)
            else:
                sig.stop_loss = round(sig.entry + risk, 2)
                sig.target    = round(sig.entry - 1.4 * risk, 2)

        # ── Multi-factor scoring ──────────────────────────────────────────
        score = 0
        vwap_against = False   # price clearly on the wrong side of VWAP
        ema_against  = False   # EMA trend clearly opposes the signal

        # 1. VWAP alignment
        if i < len(vwap) and not np.isnan(vwap[i]):
            if sig.signal == "BUY"  and closes[i] > vwap[i]: score += 1
            elif sig.signal == "SELL" and closes[i] < vwap[i]: score += 1
            else:
                score -= 1   # trading against VWAP
                vwap_against = True
                sig.above_vwap = False

        # 2. EMA 9/21 trend direction
        if (i < len(ema9) and i < len(ema21)
                and not np.isnan(ema9[i]) and not np.isnan(ema21[i])):
            if sig.signal == "BUY"  and ema9[i] > ema21[i]: score += 1
            elif sig.signal == "SELL" and ema9[i] < ema21[i]: score += 1
            else:
                ema_against = True

        # Flag hard counter-trend setups (both VWAP and EMA disagree) so the
        # caller can drop them — these are the lowest-probability scalps.
        sig.counter_trend = bool(vwap_against and ema_against)

        # 3. RSI zone (avoid entering extremes against direction)
        if i < len(rsi) and not np.isnan(rsi[i]):
            r = rsi[i]
            if sig.signal == "BUY"  and 38 <= r <= 68: score += 1
            elif sig.signal == "SELL" and 32 <= r <= 62: score += 1
            if sig.signal == "BUY"  and r > 78: score -= 1  # chasing overbought
            if sig.signal == "SELL" and r < 22: score -= 1  # chasing oversold

        # 4. Volume surge
        vol_ratio = _volume_ratio(df, i)
        if   vol_ratio >= 2.0: score += 2
        elif vol_ratio >= 1.4: score += 1
        elif vol_ratio <  0.7: score -= 1

        # 5. Time-of-day
        td = _time_score(sig.timestamp)
        if   td > 0:  score += 1
        elif td < 0:  score -= 1

        sig.confidence = round(min(0.93, max(0.38, sig.confidence + score * 0.04)), 2)

        # Recompute risk_reward
        risk   = abs(sig.entry - sig.stop_loss)
        reward = abs(sig.target - sig.entry)
        sig.risk_reward = round(reward / risk, 2) if risk > 0 else 0.0

        enhanced.append(sig)
    return enhanced


# ─── Session Regime Filter ────────────────────────────────────────────────────

def _is_bull_regime(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray,
                     ema9: np.ndarray, ema21: np.ndarray, vwap: np.ndarray,
                     atr: np.ndarray, i: int, lookback: int = 5) -> bool:
    """
    Hard pre-trade checklist — ALL 4 must pass before any BUY fires.

    Big traders don't take a trade just because a candle pattern appeared.
    They first confirm the SESSION ENVIRONMENT is favourable. This function
    answers: "Is the wind blowing in our direction RIGHT NOW?"

    Check 1 — EMA stack persistent:
        EMA9 > EMA21 for every one of the last `lookback` bars.
        A single bar cross doesn't count. We need a CONFIRMED trend.
        (eliminates all morning / choppy signals)

    Check 2 — Price above VWAP:
        Institutions accumulate above VWAP, distribute below.
        Buying below VWAP = fighting the smart money.

    Check 3 — Positive momentum (20-bar linear regression slope > 0):
        Even if EMAs are aligned today, is the price ACTUALLY moving up?
        Catches sideways days where EMAs happen to be stacked but no edge.

    Check 4 — Higher-lows structure in last 8 bars:
        Price must not have made a fresh lower low recently.
        This blocks 'catch-the-falling-knife' type early reversals —
        the Double Bottoms / Morning Stars that keep firing during a dip.
    """
    if i < max(lookback + 1, 22):
        return False

    # Check 1: EMA9 > EMA21 for last `lookback` consecutive bars
    for j in range(i - lookback + 1, i + 1):
        if np.isnan(ema9[j]) or np.isnan(ema21[j]):
            return False
        if ema9[j] <= ema21[j]:
            return False          # trend not yet established

    # Check 2: Current price above VWAP
    if not np.isnan(vwap[i]) and closes[i] < vwap[i]:
        return False

    # Check 3: Positive momentum — 20-bar linear regression slope
    y = closes[max(0, i - 19):i + 1]
    if len(y) >= 10:
        slope = np.polyfit(range(len(y)), y, 1)[0]
        if slope <= 0:
            return False          # price drifting sideways or down

    # Check 4: No fresh lower lows in last 8 bars
    #   Compare mean of first-half lows vs second-half lows —
    #   second half must NOT be lower (would mean breakdown in progress)
    win = lows[max(0, i - 7):i + 1]
    if len(win) >= 6:
        mid = len(win) // 2
        atr_v = atr[i] if (not np.isnan(atr[i]) and atr[i] > 0) else 1.0
        if np.mean(win[mid:]) < np.mean(win[:mid]) - 0.15 * atr_v:
            return False          # price structure still trending lower

    return True


# ─── Main Entry Point ─────────────────────────────────────────────────────────

def detect_all_patterns(df: pd.DataFrame, buy_only: bool = True) -> List[PatternSignal]:
    """
    Run all pattern detectors on OHLCV dataframe.
    Applies multi-factor scoring, ATR-based stops, and duplicate filtering.

    buy_only: when True (default) only long/BUY (CE) setups are returned and
              every BUY must have VWAP support (price above VWAP) — we buy
              strength, never a falling knife. SELL/PE signals are dropped.
    Returns List[PatternSignal] sorted by bar index.
    """
    if df is None or len(df) < 10:
        return []

    # Pre-compute indicators once
    closes_arr = df["close"].values.astype(float)
    vwap  = _compute_vwap(df)
    atr   = _compute_atr(df, period=14)
    ema9  = _compute_ema(closes_arr, 9)
    ema21 = _compute_ema(closes_arr, 21)
    rsi   = _compute_rsi(df, period=14)
    or_high, or_low, or_end = _opening_range(df, minutes=15)

    signals: List[PatternSignal] = []
    try:
        # ── Classic candlestick patterns ──────────────────────────────────
        signals += detect_hammer(df)
        signals += detect_shooting_star(df)
        signals += detect_inverted_hammer(df)
        signals += detect_hanging_man(df)
        signals += detect_doji(df)
        signals += detect_dragonfly_doji(df)
        signals += detect_gravestone_doji(df)
        signals += detect_engulfing(df)
        signals += detect_morning_star(df)
        signals += detect_evening_star(df)
        signals += detect_inside_bar(df)
        signals += detect_pin_bar(df)
        signals += detect_double_top(df)
        signals += detect_double_bottom(df)
        signals += detect_head_and_shoulders(df)
        signals += detect_inverse_head_and_shoulders(df)
        signals += detect_ascending_triangle(df)
        signals += detect_descending_triangle(df)
        signals += detect_bull_flag(df)
        signals += detect_bear_flag(df)

        # ── Professional scalping patterns ────────────────────────────────
        signals += detect_vwap_bounce(df, vwap, atr)
        signals += detect_orb_breakout(df, or_high, or_low, or_end, atr)
        signals += detect_ema_cross(df, ema9, ema21, vwap, atr)
        signals += detect_momentum_breakout(df, vwap, atr)
        signals += detect_rsi_reversal(df, rsi, atr, ema9, ema21)
        signals += detect_range_breakout(df, atr, vwap)
        signals += detect_ema_pullback(df, ema9, ema21, vwap, atr)

    except Exception as e:
        logger.error(f"Pattern detection error: {e}")

    # Apply multi-factor scoring & ATR-based stop widening
    signals = _apply_multi_factor(signals, df, vwap, atr, ema9, ema21, rsi)

    # Quality gates: drop low-confidence, poor-R:R, and hard counter-trend setups.
    # Trading WITH the trend (VWAP + EMA) is the single biggest edge for scalping,
    # so a signal fighting both is filtered out entirely.
    signals = [s for s in signals
               if s.confidence >= 0.58 and s.risk_reward >= 1.2 and not s.counter_trend]

    # BUY-only mode: keep long/CE setups, and require VWAP support so we only
    # buy strength (price above VWAP) instead of catching morning down-moves.
    if buy_only:
        signals = [s for s in signals if s.signal == "BUY" and s.above_vwap]

    # ── SESSION REGIME GATE (hardest filter) ─────────────────────────────────
    # A pattern is meaningless without the right environment. Every remaining
    # signal must pass all 4 regime checks (persistent EMA stack, above VWAP,
    # positive momentum, higher-lows structure).  This is what eliminates the
    # morning-dip hammers, early double-bottoms and choppy inside bars.
    h_arr = df["high"].values.astype(float)
    l_arr = df["low"].values.astype(float)
    signals = [
        s for s in signals
        if _is_bull_regime(closes_arr, h_arr, l_arr, ema9, ema21, vwap, atr, s.index)
    ]

    # Deduplicate: best confidence per index
    best: dict[int, PatternSignal] = {}
    for sig in signals:
        if sig.index not in best or sig.confidence > best[sig.index].confidence:
            best[sig.index] = sig

    # Sort by bar index
    sorted_sigs = sorted(best.values(), key=lambda x: x.index)

    # Two-pass deduplication so only clean, spaced entries reach the user:
    # Pass 1 — same PATTERN name: keep only the first occurrence per 10-bar window.
    #           This removes the same Double Bottom / Ascending Triangle appearing
    #           repeatedly as the sliding window advances.
    pattern_last: dict[str, int] = {}
    pass1: List[PatternSignal] = []
    for sig in sorted_sigs:
        last_idx = pattern_last.get(sig.pattern, -999)
        if sig.index - last_idx >= 10:
            pass1.append(sig)
            pattern_last[sig.pattern] = sig.index

    # Pass 2 — any same-direction signal within 5 bars: keep highest confidence.
    filtered: List[PatternSignal] = []
    for sig in pass1:
        if not filtered:
            filtered.append(sig)
            continue
        last = filtered[-1]
        if sig.index - last.index < 5 and sig.signal == last.signal:
            if sig.confidence > last.confidence:
                filtered[-1] = sig
        else:
            filtered.append(sig)

    return filtered
