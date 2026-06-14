"""
Live intraday chart-structure analysis.

Produces geometric overlays for the candlestick chart:
  • Diagonal trend lines (ascending support / descending resistance) fitted
    through recent swing pivots and extended to the latest bar.
  • Trend channel (parallel support+resistance) when both sides are present.
  • W pattern (double bottom)  → two equal lows + neckline.
  • M pattern (double top)     → two equal highs + neckline.

Everything is returned as index-based segments so the app can map the bar
index onto the actual timestamp axis when drawing.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def _pivots(highs: np.ndarray, lows: np.ndarray, w: int):
    """Return (pivot_high_idx, pivot_low_idx) lists using a +/- w window."""
    n = len(highs)
    ph, pl = [], []
    for i in range(w, n - w):
        if highs[i] >= highs[i - w:i + w + 1].max():
            ph.append(i)
        if lows[i] <= lows[i - w:i + w + 1].min():
            pl.append(i)
    return ph, pl


def _line_through(i0, y0, i1, y1, x_end):
    """Extend the line through (i0,y0)-(i1,y1) up to x_end. Returns (xs, ys)."""
    if i1 == i0:
        return [i0, x_end], [y0, y0]
    slope = (y1 - y0) / (i1 - i0)
    y_end = y1 + slope * (x_end - i1)
    return [i0, x_end], [y0, y_end]


def analyze_trendlines(df: pd.DataFrame, max_window: int = 4) -> dict:
    """
    Analyse an intraday OHLC DataFrame and return drawable overlays.

    Returns a dict:
      {
        "lines":       [ {x_idx,[y],color,dash,width,name} ],
        "shapes":      [ {x_idx,[y],color,name,neckline} ],
        "annotations": [ {x_idx,y,text,color} ],
      }
    All x positions are bar indices into df.
    """
    out = {"lines": [], "shapes": [], "annotations": []}
    if df is None or df.empty or len(df) < 12:
        return out

    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    n = len(df)
    last = n - 1
    price = float(closes[-1]) or 1.0
    tol = price * 0.0025          # ~0.25% equality tolerance for W/M lows/highs

    # Scale the pivot window to the series length so it works on short days.
    w = max(2, min(max_window, n // 15))
    ph, pl = _pivots(highs, lows, w)

    # ── Descending resistance trendline: last two falling pivot highs ─────────
    if len(ph) >= 2:
        a, b = ph[-2], ph[-1]
        if highs[b] <= highs[a]:        # falling / flat → resistance
            xs, ys = _line_through(a, highs[a], b, highs[b], last)
            out["lines"].append({
                "x_idx": xs, "y": ys, "color": "#ff6b6b", "dash": "solid",
                "width": 1.6, "name": "Resistance Trend",
            })

    # ── Ascending support trendline: last two rising pivot lows ───────────────
    if len(pl) >= 2:
        a, b = pl[-2], pl[-1]
        if lows[b] >= lows[a]:          # rising / flat → support
            xs, ys = _line_through(a, lows[a], b, lows[b], last)
            out["lines"].append({
                "x_idx": xs, "y": ys, "color": "#4dd2a0", "dash": "solid",
                "width": 1.6, "name": "Support Trend",
            })

    # ── W pattern (double bottom): two similar lows with a peak between ───────
    if len(pl) >= 2:
        for li in range(len(pl) - 1, 0, -1):
            l2 = pl[li]
            l1 = pl[li - 1]
            if l2 < n * 0.4:            # only flag if the second low is recent-ish
                break
            if abs(lows[l1] - lows[l2]) <= tol:
                mids = [p for p in ph if l1 < p < l2]
                if not mids:
                    continue
                peak = max(mids, key=lambda p: highs[p])
                neck = float(highs[peak])
                confirmed = price > neck
                col = "#00ff88" if confirmed else "#7CFC00"
                out["shapes"].append({
                    "x_idx": [l1, peak, l2],
                    "y": [float(lows[l1]), neck, float(lows[l2])],
                    "color": col, "name": "W (Double Bottom)", "neckline": neck,
                })
                out["lines"].append({
                    "x_idx": [l1, last], "y": [neck, neck], "color": col,
                    "dash": "dot", "width": 1.2, "name": "Neckline",
                })
                tag = "W ✔ breakout" if confirmed else "W forming"
                out["annotations"].append({
                    "x_idx": l2, "y": float(lows[l2]), "text": tag, "color": col})
                break

    # ── M pattern (double top): two similar highs with a trough between ───────
    if len(ph) >= 2:
        for hi in range(len(ph) - 1, 0, -1):
            h2 = ph[hi]
            h1 = ph[hi - 1]
            if h2 < n * 0.4:
                break
            if abs(highs[h1] - highs[h2]) <= tol:
                mids = [p for p in pl if h1 < p < h2]
                if not mids:
                    continue
                trough = min(mids, key=lambda p: lows[p])
                neck = float(lows[trough])
                confirmed = price < neck
                col = "#ff4444" if confirmed else "#ff9966"
                out["shapes"].append({
                    "x_idx": [h1, trough, h2],
                    "y": [float(highs[h1]), neck, float(highs[h2])],
                    "color": col, "name": "M (Double Top)", "neckline": neck,
                })
                out["lines"].append({
                    "x_idx": [h1, last], "y": [neck, neck], "color": col,
                    "dash": "dot", "width": 1.2, "name": "Neckline",
                })
                tag = "M ✔ breakdown" if confirmed else "M forming"
                out["annotations"].append({
                    "x_idx": h2, "y": float(highs[h2]), "text": tag, "color": col})
                break

    return out
