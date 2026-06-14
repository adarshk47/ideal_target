"""
Nifty50 Pro Trader - Streamlit App
Live chart, pattern detection, OI analysis, greeks, paper trading.
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import time

st.set_page_config(
    page_title="Nifty50 Pro Trader",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS for dark theme and smooth refresh ─────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] { display: none; }
    .main { background-color: #0e1117; }
    .block-container { padding-top: 0.5rem; padding-bottom: 0.5rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 2px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e2130;
        color: #aaa;
        border-radius: 4px 4px 0 0;
        padding: 6px 16px;
    }
    .stTabs [aria-selected="true"] { background-color: #2d3250; color: #fff; }
    .metric-card {
        background: #1e2130;
        border-radius: 8px;
        padding: 12px 16px;
        text-align: center;
        border: 1px solid #2d3250;
    }
    .metric-label { font-size: 11px; color: #888; text-transform: uppercase; }
    .metric-value { font-size: 22px; font-weight: bold; color: #fff; }
    .metric-sub { font-size: 12px; color: #aaa; }
    .bullish { color: #00ff88 !important; }
    .bearish { color: #ff4444 !important; }
    .neutral { color: #ffd700 !important; }
    .badge-high { background: #1a3a1a; color: #00ff88; border-radius: 4px; padding: 2px 6px; font-size: 11px; }
    .badge-med  { background: #3a3a1a; color: #ffd700; border-radius: 4px; padding: 2px 6px; font-size: 11px; }
    .badge-low  { background: #3a1a1a; color: #ff8888; border-radius: 4px; padding: 2px 6px; font-size: 11px; }
    .refresh-bar {
        background: #1e2130;
        border-top: 1px solid #2d3250;
        padding: 6px 12px;
        border-radius: 6px;
        font-size: 12px;
        color: #888;
        margin-top: 8px;
    }
    div[data-testid="stHorizontalBlock"] { gap: 8px; }
    iframe { border: none; }
</style>
""", unsafe_allow_html=True)

IST = pytz.timezone("Asia/Kolkata")

# ── Import modules ─────────────────────────────────────────────────────────────
try:
    from modules.angelone_client import (
        fetch_candle_data, fetch_options_chain, fetch_ltp,
        get_next_weekly_expiry, get_expiry_string, get_expiry_countdown,
        is_market_open, get_atm_strike, get_strike_range, INTERVAL_MAP,
        is_connected, get_data_source, get_client, get_last_error,
        get_options_diagnostic, fetch_candle_range, expiry_weekday_for,
        get_expiry_timeline_summary, flag_expiry_days,
        INSTRUMENT_CONFIG, fetch_ltp_for, fetch_candle_data_for,
        get_next_expiry_for, fetch_options_chain_for,
    )
    from modules.pattern_detector import detect_all_patterns
    from modules.oi_analyzer import (
        compute_delta_oi, build_oi_timeframe_table,
        get_oi_arrow_annotations, build_strike_volume_table,
        get_most_traded_strikes,
    )
    from modules.greeks_analyzer import analyze_greeks, build_greeks_trend_table, get_gamma_exposure
    from modules.black_scholes import bs_price, time_to_expiry_years, RISK_FREE_RATE as BS_RATE
    from modules.chart_analysis import analyze_trendlines
    from modules.paper_trader import (
        is_market_open as paper_market_open,
        add_paper_trade, update_paper_trades, get_trades_df,
        get_paper_trade_summary, should_add_new_trade, clear_all_trades,
        add_chart_rec_trade, get_chart_rec_trades_df,
    )
    MODULES_OK = True
except Exception as e:
    MODULES_OK = False
    st.error(f"Module import error: {e}")

# ── Autorefresh (5 seconds) ────────────────────────────────────────────────────
try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False

# ── Session state init ─────────────────────────────────────────────────────────
if "chart_tf" not in st.session_state:
    st.session_state["chart_tf"] = 5
if "last_signal_time" not in st.session_state:
    st.session_state["last_signal_time"] = None
# Per-instrument recommendation history and paper-trade state
for _inst in ("NIFTY", "SENSEX", "SBIN"):
    if f"recommendation_history_{_inst}" not in st.session_state:
        st.session_state[f"recommendation_history_{_inst}"] = []
    if f"paper_trades_{_inst}" not in st.session_state:
        st.session_state[f"paper_trades_{_inst}"] = []
    if f"paper_trade_counter_{_inst}" not in st.session_state:
        st.session_state[f"paper_trade_counter_{_inst}"] = 0
    if f"chart_rec_trades_{_inst}" not in st.session_state:
        st.session_state[f"chart_rec_trades_{_inst}"] = []
# Keep legacy key for backward compat with any existing session
if "recommendation_history" not in st.session_state:
    st.session_state["recommendation_history"] = []


def get_now():
    return datetime.now(IST)


def color_bias(bias: str) -> str:
    if bias in ("BULLISH", "BUY"):
        return "bullish"
    if bias in ("BEARISH", "SELL"):
        return "bearish"
    return "neutral"


def filter_to_latest_day(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the most recent trading day's candles (intraday view)."""
    if df is None or df.empty:
        return df
    ts = pd.to_datetime(df["timestamp"])
    last_date = ts.iloc[-1].date()
    mask = ts.dt.date == last_date
    return df[mask].reset_index(drop=True)


def style_cells(styler, func, subset):
    """Apply a cell-wise style, compatible across pandas versions."""
    if hasattr(styler, "map"):
        try:
            return styler.map(func, subset=subset)
        except TypeError:
            pass
    return styler.applymap(func, subset=subset)


def _scan_exit(candle_df: pd.DataFrame, entry_idx: int, signal_type: str,
               sl: float, target: float):
    """Scan candles forward from entry_idx to find when spot hits SL or target.

    Returns (exit_time_str | None, status_str, exit_idx) based on actual candle data.
    """
    if candle_df is None or candle_df.empty or entry_idx < 0:
        return None, "OPEN", -1
    for i in range(entry_idx + 1, len(candle_df)):
        row = candle_df.iloc[i]
        ts_str = pd.Timestamp(candle_df["timestamp"].iloc[i]).strftime("%H:%M:%S")
        if signal_type == "BUY":
            if float(row["low"]) <= sl:
                return ts_str, "LOSS", i
            if float(row["high"]) >= target:
                return ts_str, "PROFIT", i
        else:
            if float(row["high"]) >= sl:
                return ts_str, "LOSS", i
            if float(row["low"]) <= target:
                return ts_str, "PROFIT", i
    # SL/Target not hit during session
    return None, "OPEN", -1


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
def render_header(ltp: float, spot_prev: float, connected: bool):
    now = get_now()
    expiry_dt = get_next_weekly_expiry()
    expiry_str = get_expiry_string(expiry_dt)
    countdown = get_expiry_countdown(expiry_dt)
    market_status = "🟢 MARKET OPEN" if is_market_open() else "🔴 MARKET CLOSED"

    # ── Connection status badge (top-right) ──────────────────────────────────
    if connected:
        conn_html = ('<span style="background:#0d2818;color:#00ff88;border:1px solid #00ff88;'
                     'border-radius:14px;padding:4px 14px;font-size:13px;font-weight:bold;">'
                     '🟢 AngelOne · LIVE</span>')
    else:
        conn_html = ('<span style="background:#2a1010;color:#ff5555;border:1px solid #ff5555;'
                     'border-radius:14px;padding:4px 14px;font-size:13px;font-weight:bold;">'
                     '🔴 DEMO · Not connected to AngelOne</span>')
    st.markdown(
        f'<div style="display:flex;justify-content:flex-end;margin-bottom:6px;">{conn_html}</div>',
        unsafe_allow_html=True,
    )
    chg = ltp - spot_prev
    chg_pct = chg / spot_prev * 100 if spot_prev else 0
    chg_color = "#00ff88" if chg >= 0 else "#ff4444"
    chg_sign = "+" if chg >= 0 else ""
    ltp_str = f"{ltp:,.2f}" if ltp else "---"
    chg_str = f"{chg_sign}{chg:.2f} ({chg_sign}{chg_pct:.2f}%)" if ltp else "Connect to AngelOne"
    is_expiry_today = (expiry_dt is not None and expiry_dt.date() == now.date())
    expiry_day_str = "📅 TODAY!" if is_expiry_today else (
        expiry_dt.strftime("%A") if expiry_dt else "---")

    col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 2])
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">NIFTY 50</div>
            <div class="metric-value" style="color:{chg_color if ltp else '#888'};">{ltp_str}</div>
            <div class="metric-sub" style="color:{chg_color if ltp else '#666'};">{chg_str}</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">IST Time</div>
            <div class="metric-value" style="font-size:18px;">{now.strftime('%H:%M:%S')}</div>
            <div class="metric-sub">{now.strftime('%A, %d %b %Y')}</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Weekly Expiry</div>
            <div class="metric-value" style="font-size:18px; color:#ffd700;">{expiry_str}</div>
            <div class="metric-sub">{expiry_day_str}</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        cntdwn_color = "#ff4444" if is_expiry_today else "#ffd700"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Expiry Countdown</div>
            <div class="metric-value" style="font-size:16px; color:{cntdwn_color};">{countdown}</div>
            <div class="metric-sub">Time to settlement</div>
        </div>""", unsafe_allow_html=True)
    with col5:
        mkt_color = "#00ff88" if is_market_open() else "#ff4444"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Status</div>
            <div class="metric-value" style="font-size:16px; color:{mkt_color};">{market_status}</div>
            <div class="metric-sub">NSE · 09:15 - 15:30 IST</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("<hr style='border-color:#2d3250;margin:6px 0;'>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _find_support_resistance(df: pd.DataFrame, left: int = 3, right: int = 3,
                             max_levels: int = 3):
    """
    Detect swing-based support & resistance levels using local pivots,
    then merge nearby levels and return the strongest few of each.
    Returns (supports, resistances) as sorted lists of price floats.
    """
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)
    res_pivots, sup_pivots = [], []
    for i in range(left, n - right):
        win_h = highs[i - left:i + right + 1]
        win_l = lows[i - left:i + right + 1]
        if highs[i] >= win_h.max():
            res_pivots.append(highs[i])
        if lows[i] <= win_l.min():
            sup_pivots.append(lows[i])

    price = float(df["close"].iloc[-1]) or 1.0
    tol = price * 0.0012  # ~0.12% clustering tolerance

    def _cluster(levels):
        if not levels:
            return []
        levels = sorted(levels)
        clusters = [[levels[0]]]
        for lv in levels[1:]:
            if abs(lv - clusters[-1][-1]) <= tol:
                clusters[-1].append(lv)
            else:
                clusters.append([lv])
        # (avg_level, touches) — more touches = stronger
        scored = [(sum(c) / len(c), len(c)) for c in clusters]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [round(lv, 2) for lv, _ in scored[:max_levels]]

    return _cluster(sup_pivots), _cluster(res_pivots)


def _pick_active_signal(selected):
    """From the deduped (idx,(score,pat)) list pick the most recent signal."""
    if not selected:
        return None
    # selected is sorted by bar index; last is most recent
    return selected[-1][1][1]


# ─────────────────────────────────────────────────────────────────────────────
# CHART
# ─────────────────────────────────────────────────────────────────────────────
def build_chart(candle_df: pd.DataFrame, patterns, oi_annotations, tf_minutes: int,
                instrument_label: str = "NIFTY50"):
    if candle_df is None or candle_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No data — connect to AngelOne (add API secrets) to load candles",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            font_color="#fff",
            height=520,
        )
        return fig

    # Subplots: price + volume row
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.02,
        subplot_titles=["", "Volume"],
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=candle_df["timestamp"],
        open=candle_df["open"],
        high=candle_df["high"],
        low=candle_df["low"],
        close=candle_df["close"],
        name=instrument_label,
        increasing_line_color="#00ff88",
        decreasing_line_color="#ff4444",
        increasing_fillcolor="#00cc66",
        decreasing_fillcolor="#cc2222",
        whiskerwidth=0.8,
    ), row=1, col=1)

    # EMA 9 & EMA 21 overlays
    if len(candle_df) >= 2:
        ema9 = candle_df["close"].ewm(span=9, adjust=False).mean()
        ema21 = candle_df["close"].ewm(span=21, adjust=False).mean()
        fig.add_trace(go.Scatter(
            x=candle_df["timestamp"], y=ema9, mode="lines",
            line=dict(color="#ffaa00", width=1.4), name="EMA 9",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=candle_df["timestamp"], y=ema21, mode="lines",
            line=dict(color="#33b5ff", width=1.4), name="EMA 21",
        ), row=1, col=1)

    # Support / Resistance levels
    supports, resistances = _find_support_resistance(candle_df)
    x0 = candle_df["timestamp"].iloc[0]
    x1 = candle_df["timestamp"].iloc[-1]
    for lv in resistances:
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[lv, lv], mode="lines",
            line=dict(color="#ff6b6b", width=1, dash="dot"),
            name="Resistance", legendgroup="sr", showlegend=False,
            hovertemplate=f"Resistance {lv}<extra></extra>",
        ), row=1, col=1)
        fig.add_annotation(x=x1, y=lv, text=f"R {lv:.0f}", showarrow=False,
                           xanchor="left", font=dict(color="#ff6b6b", size=9),
                           row=1, col=1)
    for lv in supports:
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[lv, lv], mode="lines",
            line=dict(color="#4dd2a0", width=1, dash="dot"),
            name="Support", legendgroup="sr", showlegend=False,
            hovertemplate=f"Support {lv}<extra></extra>",
        ), row=1, col=1)
        fig.add_annotation(x=x1, y=lv, text=f"S {lv:.0f}", showarrow=False,
                           xanchor="left", font=dict(color="#4dd2a0", size=9),
                           row=1, col=1)

    # Overall direction — faint linear-regression guide over the session
    if len(candle_df) >= 3:
        y = candle_df["close"].values.astype(float)
        x_idx = np.arange(len(y))
        slope, intercept = np.polyfit(x_idx, y, 1)
        y_fit = slope * x_idx + intercept
        trend_up = slope >= 0
        fig.add_trace(go.Scatter(
            x=candle_df["timestamp"], y=y_fit, mode="lines",
            line=dict(color="#00ff88" if trend_up else "#ff4444",
                      width=1, dash="dash"),
            opacity=0.45,
            name=f"Bias ({'UP' if trend_up else 'DOWN'})",
        ), row=1, col=1)

    # ── Structural trend lines + W/M patterns (drawn on the live chart) ───────
    ts_series = candle_df["timestamp"]

    def _ts(idx):
        i = int(max(0, min(idx, len(ts_series) - 1)))
        return ts_series.iloc[i]

    try:
        struct = analyze_trendlines(candle_df)
    except Exception:
        struct = {"lines": [], "shapes": [], "annotations": []}

    for ln in struct.get("lines", []):
        fig.add_trace(go.Scatter(
            x=[_ts(i) for i in ln["x_idx"]], y=ln["y"], mode="lines",
            line=dict(color=ln["color"], width=ln.get("width", 1.4),
                      dash=ln.get("dash", "solid")),
            name=ln.get("name", "Trend"),
            hovertemplate=f"{ln.get('name','Trend')}<extra></extra>",
        ), row=1, col=1)

    for sh in struct.get("shapes", []):
        fig.add_trace(go.Scatter(
            x=[_ts(i) for i in sh["x_idx"]], y=sh["y"], mode="lines+markers",
            line=dict(color=sh["color"], width=2.2),
            marker=dict(size=8, color=sh["color"]),
            name=sh.get("name", "Pattern"),
            hovertemplate=f"{sh.get('name','Pattern')}<extra></extra>",
        ), row=1, col=1)

    for an in struct.get("annotations", []):
        fig.add_annotation(
            x=_ts(an["x_idx"]), y=an["y"], text=an["text"], showarrow=True,
            arrowhead=2, arrowcolor=an["color"], arrowsize=0.8,
            font=dict(color=an["color"], size=10, family="monospace"),
            bgcolor="rgba(0,0,0,0.65)", row=1, col=1,
        )

    # Volume bars
    colors = ["#00cc66" if c >= o else "#cc2222"
              for c, o in zip(candle_df["close"], candle_df["open"])]
    fig.add_trace(go.Bar(
        x=candle_df["timestamp"],
        y=candle_df["volume"],
        name="Volume",
        marker_color=colors,
        opacity=0.7,
        showlegend=False,
    ), row=2, col=1)

    # Pattern markers — deduped & capped so the chart stays readable.
    # Keep only the highest-confidence pattern per bar, then show the most
    # recent ones (clutter from 40+ overlapping labels otherwise).
    def _conf_score(p):
        c = getattr(p, "confidence", 0.5)
        if isinstance(c, str):
            return {"HIGH": 1.0, "MEDIUM": 0.6, "LOW": 0.3}.get(c.upper(), 0.5)
        return float(c)

    best_per_bar = {}
    for pat in patterns:
        idx = getattr(pat, "index", getattr(pat, "bar_index", -1))
        if idx < 0 or idx >= len(candle_df):
            continue
        score = _conf_score(pat) + min(float(pat.risk_reward or 0) / 10, 0.5)
        if idx not in best_per_bar or score > best_per_bar[idx][0]:
            best_per_bar[idx] = (score, pat)

    # Most recent 5 bars with a pattern (sorted by bar index) — keeps chart readable
    selected = sorted(best_per_bar.items())[-5:]

    buy_x, buy_y, buy_text = [], [], []
    sell_x, sell_y, sell_text = [], [], []
    for idx, (_, pat) in selected:
        ts = candle_df["timestamp"].iloc[idx]
        pname = getattr(pat, "pattern", getattr(pat, "name", str(pat)))
        label = f"{pname} (RR {pat.risk_reward})"
        # Stagger label offset using ATR-like spacing to avoid overlap
        rng = float(candle_df["high"].iloc[idx] - candle_df["low"].iloc[idx]) or 5
        if pat.signal == "BUY":
            buy_x.append(ts)
            buy_y.append(candle_df["low"].iloc[idx] - rng * 0.8)
            buy_text.append(label)
        else:
            sell_x.append(ts)
            sell_y.append(candle_df["high"].iloc[idx] + rng * 0.8)
            sell_text.append(label)

    if buy_x:
        fig.add_trace(go.Scatter(
            x=buy_x, y=buy_y, mode="markers+text",
            marker=dict(symbol="triangle-up", size=14, color="#00ff88"),
            text=buy_text, textposition="bottom center",
            textfont=dict(size=9, color="#00ff88"),
            name="BUY Signal", showlegend=True,
        ), row=1, col=1)

    if sell_x:
        fig.add_trace(go.Scatter(
            x=sell_x, y=sell_y, mode="markers+text",
            marker=dict(symbol="triangle-down", size=14, color="#ff4444"),
            text=sell_text, textposition="top center",
            textfont=dict(size=9, color="#ff4444"),
            name="SELL Signal", showlegend=True,
        ), row=1, col=1)

    # ── Entry / Stop-Loss / Target levels for the most recent signal ─────────
    active = _pick_active_signal(selected)
    if active is not None:
        is_buy = active.signal == "BUY"
        side_color = "#00ff88" if is_buy else "#ff4444"
        levels = [
            ("ENTRY", float(active.entry), "#ffffff"),
            ("SL", float(active.stop_loss), "#ff5555"),
            ("TARGET", float(active.target), "#00ff88"),
        ]
        for label, price_lv, lvl_color in levels:
            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[price_lv, price_lv], mode="lines",
                line=dict(color=lvl_color, width=1.3,
                          dash="solid" if label == "ENTRY" else "dashdot"),
                name=f"{label}", showlegend=False,
                hovertemplate=f"{label} {price_lv:.2f}<extra></extra>",
            ), row=1, col=1)
            fig.add_annotation(
                x=x1, y=price_lv, text=f"{label} {price_lv:.0f}",
                showarrow=False, xanchor="left",
                bgcolor="rgba(0,0,0,0.6)",
                font=dict(color=lvl_color, size=10, family="monospace"),
                row=1, col=1,
            )
        # Headline badge for the active trade
        pname = getattr(active, "pattern", getattr(active, "name", "Signal"))
        fig.add_annotation(
            x=x0, y=candle_df["high"].max(),
            text=f"▶ {active.signal} · {pname} · R:R 1:{active.risk_reward}",
            showarrow=False, xanchor="left", yanchor="top",
            bgcolor=side_color, font=dict(color="#0e1117", size=11),
            row=1, col=1,
        )

    # Layout
    fig.update_layout(
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="#ccc", size=11),
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            bgcolor="rgba(0,0,0,0.5)",
        ),
        margin=dict(l=10, r=10, t=30, b=10),
        height=520,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        title=dict(text=f"{instrument_label} | {tf_minutes}min Chart", font=dict(size=14, color="#fff")),
    )
    fig.update_xaxes(
        gridcolor="#1e2130", showgrid=True,
        tickformat="%H:%M" if tf_minutes <= 15 else "%d%b %H:%M",
        rangeslider_visible=False,
    )
    fig.update_yaxes(gridcolor="#1e2130", showgrid=True)

    # OI arrows — append (do NOT replace existing S/R & entry/target labels)
    if oi_annotations:
        for ann in oi_annotations:
            try:
                fig.add_annotation(ann)
            except Exception:
                pass

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

def render_recommendations_tab(patterns, spot: float, instrument: str = "NIFTY"):
    st.markdown("### 📋 Pattern Recommendations – Today's History")
    market_open = is_market_open()
    hist_key = f"recommendation_history_{instrument}"

    if not market_open:
        st.warning("⚠️ Market is closed. No new recommendations. Showing historical data only.")

    if market_open and patterns:
        for pat in patterns:
            pat_name = getattr(pat, "pattern", getattr(pat, "name", "Signal"))
            ts_key = f"{pat_name}_{pat.signal}_{pat.entry}"
            if ts_key not in [r.get("key") for r in st.session_state[hist_key]]:
                confidence = getattr(pat, "confidence", 0.5)
                if isinstance(confidence, (int, float)):
                    conf_str = "HIGH" if confidence > 0.7 else "MEDIUM" if confidence > 0.4 else "LOW"
                else:
                    conf_str = str(confidence)
                st.session_state[hist_key].append({
                    "key": ts_key,
                    "time": datetime.now(IST).strftime("%H:%M:%S"),
                    "pattern": pat_name,
                    "signal": pat.signal,
                    "entry": pat.entry,
                    "sl": pat.stop_loss,
                    "target": pat.target,
                    "rr": pat.risk_reward,
                    "confidence": conf_str,
                    "description": getattr(pat, "description", ""),
                })

    if not st.session_state[hist_key]:
        st.info("No patterns detected yet. Waiting for market data...")
        return

    for rec in reversed(st.session_state[hist_key][-20:]):
        sig_color = "#00ff88" if rec["signal"] == "BUY" else "#ff4444"
        conf = rec.get("confidence", "MEDIUM")
        badge_cls = "badge-high" if conf == "HIGH" else "badge-med" if conf == "MEDIUM" else "badge-low"
        st.markdown(f"""
        <div style="background:#1e2130;border-left:3px solid {sig_color};padding:8px 12px;
                    border-radius:0 6px 6px 0;margin-bottom:6px;">
            <span style="color:{sig_color};font-weight:bold;">{rec['signal']}</span>
            &nbsp;|&nbsp;<b>{rec['pattern']}</b>
            &nbsp;<span class="{badge_cls}">{conf}</span>
            &nbsp;&nbsp;<span style="color:#888;font-size:12px;">{rec['time']}</span><br>
            <span style="font-size:12px;color:#aaa;">
                Entry: <b style="color:#fff;">{rec['entry']}</b> &nbsp;
                SL: <b style="color:#ff8888;">{rec['sl']}</b> &nbsp;
                Target: <b style="color:#88ff88;">{rec['target']}</b> &nbsp;
                R:R <b style="color:#ffd700;">1:{rec['rr']}</b>
            </span><br>
            <span style="font-size:11px;color:#666;">{rec.get('description','')}</span>
        </div>
        """, unsafe_allow_html=True)

    if st.button("🗑️ Clear History", key=f"clear_rec_history_{instrument}"):
        st.session_state[hist_key] = []
        st.rerun()


def render_strike_volume_tab(options_df: pd.DataFrame, spot: float, candle_data_by_tf: dict,
                             instrument: str = "NIFTY"):
    st.markdown("### 🎯 Most Traded Strikes – ATM ±5")
    if options_df is None or options_df.empty:
        st.warning(f"Options data unavailable — {get_options_diagnostic()}")
        return

    tf_col, _ = st.columns([1, 3])
    with tf_col:
        selected_tf = st.selectbox("Timeframe", [1, 2, 5, 10, 15, 30, 60], index=2,
                                   key=f"strike_tf_select_{instrument}",
                                   format_func=lambda x: f"{x} min")

    strike_df = build_strike_volume_table(options_df, candle_data_by_tf, spot, n_strikes=5)
    if not strike_df.empty:
        most_traded = get_most_traded_strikes(options_df, spot, n_strikes=5)
        if not most_traded.empty:
            top_strike = most_traded.iloc[0]
            atm = round(spot / 50) * 50
            st.markdown(f"""
            <div style="background:#1a2a1a;border-radius:8px;padding:10px 16px;margin-bottom:12px;">
                🔥 <b>Most Active Strike: {int(top_strike['strike'])}</b> &nbsp;|&nbsp;
                Total Vol: <b>{int(top_strike['total_volume']):,}</b> &nbsp;|&nbsp;
                PCR: <b>{top_strike['ce_pe_ratio']:.2f}</b> &nbsp;|&nbsp;
                ATM: <b>{int(atm)}</b>
            </div>""", unsafe_allow_html=True)

        fmt = {}
        for c in strike_df.columns:
            if c in ("CE LTP", "PE LTP", "PCR"):
                fmt[c] = "{:.2f}"
            elif c == "Strike":
                fmt[c] = "{:.0f}"
            elif c in ("CE OI", "PE OI", "Net OI", "CE Vol", "PE Vol", "Total Vol"):
                fmt[c] = "{:,.0f}"
        styled_strike = strike_df.style.format(fmt) if fmt else strike_df.style
        st.dataframe(styled_strike, use_container_width=True, hide_index=True)
    else:
        st.info("Loading strike data...")


def render_paper_trade_tab(patterns, spot: float, options_df=None, candle_df=None,
                           instrument: str = "NIFTY"):
    st.markdown("### 📝 Paper Trading – Auto Signals")
    market_open = is_market_open()
    effective_spot = spot if spot and spot > 0 else st.session_state.get("_last_ltp", 22000.0)

    if not market_open:
        st.warning("🔴 Market Closed — showing **simulation** results based on chart patterns "
                   "detected from last session data.")

    # Auto-add trades from patterns.
    # Live mode: OPEN trades tracked in real-time via update_paper_trades().
    # Closed mode: resolve immediately using candle-scan exit + option premium entry.
    # Strike gap per instrument
    cfg = INSTRUMENT_CONFIG.get(instrument, INSTRUMENT_CONFIG["NIFTY"])
    strike_gap = cfg.get("strike_gap", 50)

    # ATM IV from the current options snapshot (used as a flat vol for B-S pricing).
    cur_atm = float(round(effective_spot / strike_gap) * strike_gap)
    _atm_ce_iv, _atm_pe_iv = 0.15, 0.15
    if options_df is not None and not options_df.empty:
        try:
            atm_row = options_df[(options_df["strike"] - cur_atm).abs() < strike_gap * 0.6]
            if not atm_row.empty:
                _atm_ce_iv = max(float(atm_row.iloc[0].get("ce_iv", 15) or 15) / 100, 0.05)
                _atm_pe_iv = max(float(atm_row.iloc[0].get("pe_iv", 15) or 15) / 100, 0.05)
        except Exception:
            pass

    expiry_dt_for_bs = get_next_expiry_for(instrument)

    def _bs_option_price(spot_val, strike, opt_type, candle_ts):
        """B-S option price for a given strike at a given spot & candle time."""
        try:
            ts_dt = pd.Timestamp(candle_ts).to_pydatetime()
            if ts_dt.tzinfo is None:
                ts_dt = IST.localize(ts_dt)
            T = time_to_expiry_years(expiry_dt_for_bs, ts_dt)
            if T <= 0 or expiry_dt_for_bs is None:
                return 0.0
            iv = _atm_ce_iv if opt_type == "C" else _atm_pe_iv
            price = bs_price(spot_val, strike, T, BS_RATE, iv, opt_type)
            return round(max(price, 0.01), 2)
        except Exception:
            return 0.0

    if patterns:
        for pat in patterns:
            pat_name = getattr(pat, "pattern", getattr(pat, "name", "Signal"))
            sim = not market_open
            if should_add_new_trade(pat_name, pat.signal, simulated=sim, instrument=instrument):
                opt_type = "C" if pat.signal == "BUY" else "P"
                entry_idx = getattr(pat, "index", getattr(pat, "bar_index", -1))

                option_ltp = 0.0
                option_sl = None
                option_target = None
                entry_time = None
                trade_spot = effective_spot
                trade_strike = cur_atm

                if sim and candle_df is not None and not candle_df.empty and 0 <= entry_idx < len(candle_df):
                    try:
                        entry_ts = candle_df["timestamp"].iloc[entry_idx]
                        entry_time = pd.Timestamp(entry_ts).strftime("%H:%M:%S")
                        spot_at_entry = float(candle_df["close"].iloc[entry_idx])
                        trade_spot = spot_at_entry
                        trade_strike = float(round(spot_at_entry / strike_gap) * strike_gap)
                        option_ltp = _bs_option_price(spot_at_entry, trade_strike, opt_type, entry_ts)
                        option_sl = _bs_option_price(float(pat.stop_loss), trade_strike, opt_type, entry_ts)
                        option_target = _bs_option_price(float(pat.target), trade_strike, opt_type, entry_ts)
                    except Exception:
                        pass
                elif not sim and options_df is not None and not options_df.empty:
                    opt_col = "ce_ltp" if pat.signal == "BUY" else "pe_ltp"
                    try:
                        closest_idx = (options_df["strike"] - cur_atm).abs().argsort().iloc[0]
                        val = options_df.iloc[closest_idx].get(opt_col, 0.0)
                        option_ltp = float(val) if val else 0.0
                    except Exception:
                        pass

                # Scan candles for exit
                exit_info = None
                chart_exit_ts, chart_exit_status = None, "OPEN"
                if sim and candle_df is not None and not candle_df.empty:
                    exit_ts, exit_status, exit_idx = _scan_exit(
                        candle_df, entry_idx, pat.signal,
                        float(pat.stop_loss), float(pat.target),
                    )
                    chart_exit_ts, chart_exit_status = exit_ts, exit_status
                    if exit_status in ("PROFIT", "LOSS"):
                        exit_option_price = None
                        if exit_idx >= 0 and option_ltp > 0:
                            try:
                                exit_candle_ts = candle_df["timestamp"].iloc[exit_idx]
                                # Price the exit at the SPOT LEVEL THAT WAS HIT
                                # (stop-loss on a LOSS, target on a PROFIT) — not
                                # the candle close, which can wick to SL then close
                                # back the other way and flip the option P&L sign.
                                exit_spot = (float(pat.stop_loss) if exit_status == "LOSS"
                                             else float(pat.target))
                                exit_option_price = _bs_option_price(
                                    exit_spot, trade_strike, opt_type, exit_candle_ts)
                            except Exception:
                                pass
                        exit_info = {
                            "status": exit_status,
                            "exit_time": exit_ts,
                            "exit_option_price": exit_option_price,
                        }

                add_paper_trade(pat, pat_name, trade_spot,
                                source="AUTO", simulated=sim,
                                option_ltp=option_ltp, exit_info=exit_info,
                                entry_time=entry_time, strike=trade_strike,
                                option_sl=option_sl, option_target=option_target,
                                instrument=instrument)

                # Chart rec trade (spot-level, separate table)
                add_chart_rec_trade(
                    pattern_name=pat_name, signal=pat.signal,
                    entry=float(pat.entry), sl=float(pat.stop_loss),
                    target=float(pat.target), rr=pat.risk_reward,
                    entry_time=entry_time or "",
                    exit_time=chart_exit_ts or "",
                    status=chart_exit_status,
                    instrument=instrument,
                )

    # Update open trade statuses
    update_paper_trades(spot, instrument=instrument)

    # Summary
    summary = get_paper_trade_summary(instrument=instrument)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Total Trades", summary["total"])
    with c2:
        st.metric("Open", summary["open"])
    with c3:
        st.metric("Profits", summary["profit"], delta=f"+{summary['profit']}")
    with c4:
        st.metric("Losses", summary["loss"])
    with c5:
        st.metric("Total P&L", f"₹{summary['total_pnl']:+.0f}", delta=f"{summary['win_rate']:.0f}% win")

    df = get_trades_df(instrument=instrument)
    if df.empty:
        st.info("Waiting for pattern signals to initiate paper trades...")
    else:
        # ── Trade detail cards ────────────────────────────────────────────────
        st.markdown("#### 📑 Trade Details")
        for trade in reversed(st.session_state[f"paper_trades_{instrument}"][-12:]):
            status = trade["status"]
            sig_color = "#00ff88" if trade["signal"] == "BUY" else "#ff4444"
            if status == "PROFIT":
                st_color, st_icon = "#00ff88", "✅ TARGET HIT"
            elif status == "LOSS":
                st_color, st_icon = "#ff4444", "🛑 SL HIT"
            else:
                st_color, st_icon = "#aaaaff", "⏳ OPEN"
            exit_card = ""
            if trade["exit_price"] is not None:
                exit_card = (f"Exit: <b style='color:#fff;'>{float(trade['exit_price']):.2f}</b> "
                             f"@ {trade['exit_time'] or '—'} &nbsp;|&nbsp; "
                             f"P&L: <b style='color:{st_color};'>{trade['pnl']:+.2f} "
                             f"({trade['pnl_pct']:+.2f}%)</b>")
            src_badge = "SIM" if trade.get("source") == "SIM" else "LIVE"
            entry_note = " <span style='color:#888;font-size:10px;'>(premium)</span>" \
                if trade.get("source") == "SIM" and trade.get("entry_spot", 0) != trade.get("entry", 0) else ""
            st.markdown(f"""
            <div style="background:#1e2130;border-left:4px solid {st_color};
                        padding:10px 14px;border-radius:0 6px 6px 0;margin-bottom:8px;">
                <div style="display:flex;justify-content:space-between;">
                    <span><b style="color:{sig_color};">{trade['signal']}</b>
                        &nbsp;<b style="color:#fff;">{trade['option']}</b>
                        &nbsp;<span style="color:#888;font-size:11px;">[{src_badge}]</span></span>
                    <span style="color:{st_color};font-weight:bold;">{st_icon}</span>
                </div>
                <div style="font-size:12px;color:#aaa;margin-top:4px;">
                    Pattern: <b style="color:#ddd;">{trade['pattern']}</b>
                    &nbsp;({trade.get('confidence','')}) &nbsp;|&nbsp;
                    {trade['time']} {trade['date']}
                    &nbsp;|&nbsp; Spot: {float(trade.get('entry_spot', 0)):.2f}
                </div>
                <div style="font-size:13px;color:#ccc;margin-top:6px;">
                    Entry:{entry_note} <b style="color:#fff;">{float(trade['entry']):.2f}</b> &nbsp;|&nbsp;
                    SL: <b style="color:#ff8888;">{float(trade['stop_loss']):.2f}</b> &nbsp;|&nbsp;
                    Target: <b style="color:#88ff88;">{float(trade['target']):.2f}</b> &nbsp;|&nbsp;
                    R:R <b style="color:#ffd700;">1:{trade['rr']}</b>
                </div>
                <div style="font-size:13px;color:#ccc;margin-top:4px;">{exit_card}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### 📋 All Trades Table")

        def style_status(val):
            if val == "PROFIT":
                return "background-color: #1a3a1a; color: #00ff88"
            elif val == "LOSS":
                return "background-color: #3a1a1a; color: #ff4444"
            elif val == "OPEN":
                return "background-color: #1a1a3a; color: #aaaaff"
            return ""

        display_cols = ["id", "time", "pattern", "signal", "option", "entry", "stop_loss",
                        "target", "rr", "status", "exit_price", "exit_time", "pnl", "pnl_pct",
                        "confidence"]
        available = [c for c in display_cols if c in df.columns]
        styled = style_cells(df[available].style, style_status,
                             ["status"] if "status" in available else [])
        num_fmt = {c: "{:.2f}" for c in ["entry", "stop_loss", "target", "exit_price",
                                         "pnl", "pnl_pct", "rr"] if c in available}
        if num_fmt:
            styled = styled.format(num_fmt, na_rep="—")
        st.dataframe(styled, use_container_width=True, hide_index=True)

    # ── Chart Recommendation Trades (spot-level) ──────────────────────────────
    st.markdown("---")
    st.markdown("### 📋 Chart Recommendation Trades (Spot Level)")
    st.caption("Spot-price signal with entry/exit as found by candle scan — "
               "no option premium, pure price-action.")
    rec_df = get_chart_rec_trades_df(instrument=instrument)
    if rec_df.empty:
        st.info("No chart recommendation trades recorded yet.")
    else:
        def style_rec_status(val):
            if val == "PROFIT":
                return "background-color: #1a3a1a; color: #00ff88"
            elif val == "LOSS":
                return "background-color: #3a1a1a; color: #ff4444"
            return "color: #aaaaff"

        rec_styled = style_cells(rec_df.style, style_rec_status,
                                 ["status"] if "status" in rec_df.columns else [])
        num_cols = ["entry", "stop_loss", "target"]
        rec_fmt = {c: "{:.2f}" for c in num_cols if c in rec_df.columns}
        if rec_fmt:
            rec_styled = rec_styled.format(rec_fmt)
        st.dataframe(rec_styled, use_container_width=True, hide_index=True)

    if st.button("🗑️ Clear Paper Trades", key=f"clear_paper_{instrument}"):
        clear_all_trades(instrument=instrument)
        st.rerun()


def render_oi_table_tab(candle_data_by_tf: dict, options_df: pd.DataFrame, spot: float):
    st.markdown("### 📊 OI Difference Table – Trend Direction by Timeframe")
    if options_df is None or options_df.empty:
        st.warning(f"Options data unavailable — {get_options_diagnostic()}")
        return

    oi_table = build_oi_timeframe_table(candle_data_by_tf, options_df, spot)

    if oi_table.empty:
        st.info("Computing OI trends...")
        return

    def style_trend(val):
        if val == "BULLISH":
            return "background-color: #1a3a1a; color: #00ff88; font-weight: bold"
        elif val == "BEARISH":
            return "background-color: #3a1a1a; color: #ff4444; font-weight: bold"
        return "color: #ffd700"

    def style_arrow(val):
        if val == "↑":
            return "color: #00ff88; font-size: 18px; font-weight: bold"
        elif val == "↓":
            return "color: #ff4444; font-size: 18px; font-weight: bold"
        return "color: #ffd700"

    styled = style_cells(oi_table.style, style_trend, ["Trend"])
    styled = style_cells(styled, style_arrow, ["Arrow"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    delta_info = compute_delta_oi(options_df, spot)
    bias = delta_info.get("bias", "NEUTRAL")
    pcr = delta_info.get("pcr", 1.0)
    bias_color = "#00ff88" if bias == "BULLISH" else "#ff4444" if bias == "BEARISH" else "#ffd700"
    st.markdown(f"""
    <div style="background:#1e2130;border-radius:8px;padding:10px 16px;margin-top:12px;">
        <b>Current OI Bias:</b>
        <span style="color:{bias_color};font-size:18px;font-weight:bold;"> {bias}</span>
        &nbsp;|&nbsp; PCR: <b style="color:#ffd700;">{pcr:.2f}</b>
        &nbsp;|&nbsp; CE OI: <b>{delta_info.get('ce_oi',0):,}</b>
        &nbsp;|&nbsp; PE OI: <b>{delta_info.get('pe_oi',0):,}</b>
        <br><small style="color:#888;">PCR &gt; 1.2 = Oversold (Bullish) | PCR &lt; 0.8 = Overbought (Bearish)</small>
    </div>""", unsafe_allow_html=True)


def render_greeks_tab(options_df: pd.DataFrame, spot: float):
    st.markdown("### 🔢 Greeks Analysis – Gamma, Theta, Premium Trend")
    if options_df is None or options_df.empty:
        st.warning(f"Options data unavailable — {get_options_diagnostic()}")
        return

    result = analyze_greeks(options_df, spot, n_strikes=5)
    summary = result.get("summary", {})
    bias = result.get("bias", "NEUTRAL")
    table = result.get("table", pd.DataFrame())

    bias_color = "#00ff88" if bias == "BULLISH" else "#ff4444" if bias == "BEARISH" else "#ffd700"

    # Summary cards
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Greeks Bias</div>
            <div class="metric-value" style="color:{bias_color};">{bias}</div>
            <div class="metric-sub">{summary.get('Premium Skew','')}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Gamma Signal</div>
            <div class="metric-value" style="font-size:14px;">{summary.get('Gamma Signal','')}</div>
            <div class="metric-sub">ATM Γ: {summary.get('ATM Gamma',0)}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Theta Signal</div>
            <div class="metric-value" style="font-size:14px;">{summary.get('Theta Signal','')}</div>
            <div class="metric-sub">ATM Θ CE: {summary.get('ATM Theta (CE)',0)}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("#### Greeks by Strike (ATM ±5)")
    if not table.empty:
        st.dataframe(table, use_container_width=True, hide_index=True)

    st.markdown("#### Premium Trend Analysis")
    trend_table = build_greeks_trend_table(options_df, spot)
    if not trend_table.empty:
        def style_signal(val):
            if val == "BULLISH":
                return "color: #00ff88; font-weight: bold"
            elif val == "BEARISH":
                return "color: #ff4444; font-weight: bold"
            return ""

        styled = style_cells(trend_table.style, style_signal, ["Signal"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

    st.markdown("#### Gamma Exposure (GEX)")
    gex_df = get_gamma_exposure(options_df, spot)
    if not gex_df.empty:
        st.dataframe(gex_df, use_container_width=True, hide_index=True)


def _patterns_to_df(patterns) -> pd.DataFrame:
    """Convert detected pattern objects into an exportable DataFrame."""
    rows = []
    for p in patterns:
        rows.append({
            "pattern": getattr(p, "pattern", getattr(p, "name", "")),
            "signal": getattr(p, "signal", ""),
            "entry": round(float(getattr(p, "entry", 0) or 0), 2),
            "stop_loss": round(float(getattr(p, "stop_loss", 0) or 0), 2),
            "target": round(float(getattr(p, "target", 0) or 0), 2),
            "rr": getattr(p, "risk_reward", ""),
            "confidence": getattr(p, "confidence", ""),
            "bar_index": getattr(p, "index", getattr(p, "bar_index", -1)),
            "description": getattr(p, "description", ""),
        })
    return pd.DataFrame(rows)


def render_intraday_expiry_section(expiry_dt, instrument: str = "NIFTY"):
    """
    Fetch 5-min candles for a user date range from AngelOne, detect the
    expiry days inside it, render each expiry day's intraday chart + detected
    chart patterns, and let the user export the candle data and the patterns.
    """
    st.markdown("#### 🕔 Expiry-Day Intraday (5-min) Chart, Pattern & Export")
    if not is_connected():
        st.info("🔌 Connect to AngelOne (top of page) to fetch intraday 5-min history.")
        return

    k = instrument  # short alias for session state keys
    today = get_now().date()
    cda, cdb, cdc, cdd = st.columns([2, 2, 1.5, 1.5])
    with cda:
        d_from = st.date_input("From date", value=today - timedelta(days=90),
                               max_value=today, key=f"exp_intraday_from_{k}")
    with cdb:
        d_to = st.date_input("To date", value=today, max_value=today,
                             key=f"exp_intraday_to_{k}")
    with cdc:
        tf_choice = st.selectbox("Interval", [5, 1, 2, 3, 10, 15],
                                 index=0, key=f"exp_intraday_tf_{k}",
                                 format_func=lambda x: f"{x} min")
    with cdd:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        fetch_clicked = st.button("⬇️ Fetch", type="primary",
                                  use_container_width=True, key=f"exp_intraday_fetch_{k}")

    if fetch_clicked:
        with st.spinner(f"Fetching {tf_choice}-min {instrument} {d_from} → {d_to} (chunked)…"):
            rng_df = fetch_candle_range(tf_choice, d_from.strftime("%Y-%m-%d"),
                                        d_to.strftime("%Y-%m-%d"))
        if rng_df is None or rng_df.empty:
            st.error("No data returned. Check the date range / connection. "
                     "5-min history is capped at 100 days per request (auto-chunked).")
        else:
            st.session_state[f"intraday_range_df_{k}"] = rng_df
            st.session_state[f"intraday_range_tf_{k}"] = tf_choice

    rng_df = st.session_state.get(f"intraday_range_df_{k}")
    if rng_df is None or rng_df.empty:
        return

    tf_used = st.session_state.get(f"intraday_range_tf_{k}", 5)
    rng_df = rng_df.copy()
    rng_df["d"] = pd.to_datetime(rng_df["timestamp"]).dt.date

    # Identify expiry days inside the fetched range
    unique_dates = pd.Series(sorted(rng_df["d"].unique()))
    udt = pd.to_datetime(unique_dates)
    exp_flags = flag_expiry_days(udt).values
    expiry_dates = [d for d, f in zip(unique_dates, exp_flags) if f]

    st.success(f"Loaded **{len(rng_df):,}** {tf_used}-min candles over "
               f"**{len(unique_dates)}** trading days · "
               f"**{len(expiry_dates)}** expiry days detected.")

    # Full-range CSV export
    csv_all = rng_df.drop(columns=["d"]).to_csv(index=False).encode()
    st.download_button("📥 Export full range (CSV)", csv_all,
                       file_name=f"{instrument.lower()}_{tf_used}min_{d_from}_{d_to}.csv",
                       mime="text/csv", key=f"dl_range_all_{k}")

    if not expiry_dates:
        st.info("No expiry days fall inside this range.")
        return

    sel = st.selectbox(
        "Select an expiry day to view its intraday chart & pattern",
        options=list(reversed(expiry_dates)),
        format_func=lambda d: f"{d.strftime('%d %b %Y (%a)')}  ·  expiry",
        key=f"exp_intraday_day_{k}",
    )
    day_df = rng_df[rng_df["d"] == sel].drop(columns=["d"]).reset_index(drop=True)
    if day_df.empty or len(day_df) < 3:
        st.warning("Not enough candles for the selected expiry day.")
        return

    # Detect patterns on the expiry day's intraday candles
    try:
        day_patterns = detect_all_patterns(day_df)
    except Exception as e:
        day_patterns = []
        st.warning(f"Pattern detection error: {e}")

    # Day OHLC summary
    o = float(day_df["open"].iloc[0]); c = float(day_df["close"].iloc[-1])
    h = float(day_df["high"].max()); lo = float(day_df["low"].min())
    chg = (c - o) / o * 100 if o else 0
    rng_pct = (h - lo) / o * 100 if o else 0
    cc = "#00ff88" if chg >= 0 else "#ff4444"
    s1, s2, s3, s4, s5 = st.columns(5)
    for col, lbl, vl, sub, clr in [
        (s1, "Open", f"{o:,.2f}", "", "#fff"),
        (s2, "Close", f"{c:,.2f}", "", "#fff"),
        (s3, "Day Move", f"{chg:+.2f}%", "open→close", cc),
        (s4, "Range", f"{rng_pct:.2f}%", f"{lo:,.0f}–{h:,.0f}", "#ffd700"),
        (s5, "Patterns", f"{len(day_patterns)}", "detected", "#aaaaff"),
    ]:
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">{lbl}</div>
                <div class="metric-value" style="font-size:18px;color:{clr};">{vl}</div>
                <div class="metric-sub">{sub}</div></div>""", unsafe_allow_html=True)

    # Intraday chart with pattern markers
    fig_day = build_chart(day_df, day_patterns, [], tf_used)
    fig_day.update_layout(title=dict(
        text=f"{instrument} {tf_used}-min — Expiry {sel.strftime('%d %b %Y')}",
        font=dict(size=14, color="#fff")))
    st.plotly_chart(fig_day, use_container_width=True,
                    config={"displaylogo": False})

    # Pattern table + exports
    pat_df = _patterns_to_df(day_patterns)
    if not pat_df.empty:
        st.markdown("**Detected chart patterns on this expiry day:**")
        st.dataframe(pat_df, use_container_width=True, hide_index=True)
    else:
        st.caption("No high-confidence chart patterns detected on this expiry day.")

    e1, e2 = st.columns(2)
    with e1:
        st.download_button(
            "📥 Export this day's candles (CSV)",
            day_df.to_csv(index=False).encode(),
            file_name=f"nifty_{tf_used}min_expiry_{sel}.csv",
            mime="text/csv", key=f"dl_day_candles_{k}")
    with e2:
        if not pat_df.empty:
            st.download_button(
                "📥 Export this day's patterns (CSV)",
                pat_df.to_csv(index=False).encode(),
                file_name=f"{instrument.lower()}_patterns_expiry_{sel}.csv",
                mime="text/csv", key=f"dl_day_patterns_{k}")

    # All-expiry-days summary export across the whole fetched range
    with st.expander("📦 Export ALL expiry days in range (summary + patterns)"):
        summ_rows, all_pat_rows = [], []
        for ed in expiry_dates:
            edf = rng_df[rng_df["d"] == ed].drop(columns=["d"]).reset_index(drop=True)
            if len(edf) < 3:
                continue
            eo = float(edf["open"].iloc[0]); ec = float(edf["close"].iloc[-1])
            eh = float(edf["high"].max()); el = float(edf["low"].min())
            try:
                eps = detect_all_patterns(edf)
            except Exception:
                eps = []
            summ_rows.append({
                "expiry_date": ed.strftime("%Y-%m-%d"),
                "weekday": ed.strftime("%a"),
                "open": round(eo, 2), "high": round(eh, 2),
                "low": round(el, 2), "close": round(ec, 2),
                "chg_pct": round((ec - eo) / eo * 100, 2) if eo else 0,
                "range_pct": round((eh - el) / eo * 100, 2) if eo else 0,
                "n_patterns": len(eps),
                "top_pattern": (getattr(eps[0], "pattern",
                                getattr(eps[0], "name", "")) if eps else ""),
            })
            pdf = _patterns_to_df(eps)
            if not pdf.empty:
                pdf.insert(0, "expiry_date", ed.strftime("%Y-%m-%d"))
                all_pat_rows.append(pdf)
        summ_df = pd.DataFrame(summ_rows)
        if not summ_df.empty:
            st.dataframe(summ_df, use_container_width=True, hide_index=True)
            st.download_button(
                "📥 Export expiry-day summary (CSV)",
                summ_df.to_csv(index=False).encode(),
                file_name=f"nifty_expiry_summary_{d_from}_{d_to}.csv",
                mime="text/csv", key=f"dl_exp_summary_{k}")
            if all_pat_rows:
                all_pat = pd.concat(all_pat_rows, ignore_index=True)
                st.download_button(
                    "📥 Export ALL expiry-day patterns (CSV)",
                    all_pat.to_csv(index=False).encode(),
                    file_name=f"{instrument.lower()}_expiry_patterns_{d_from}_{d_to}.csv",
                    mime="text/csv", key=f"dl_exp_all_patterns_{k}")

    st.markdown("<hr style='border-color:#2d3250;margin:10px 0;'>", unsafe_allow_html=True)


def render_expiry_analysis_tab(wide_candle_df: pd.DataFrame, spot: float, expiry_dt,
                               instrument: str = "NIFTY"):
    """
    Expiry-day analysis + current expiry prediction.
    Two parts:
      1) Live 5-min intraday fetch (date→date) with per-expiry-day chart,
         pattern detection, and CSV export.
      2) Upload multi-year daily NIFTY data → profile all past expiry days
         (Thursday→Tuesday timeline aware) → predict the current expiry.
    """
    st.markdown("### 📅 Expiry Day Analysis & Prediction")

    now = get_now()
    is_expiry_today = (expiry_dt is not None and expiry_dt.date() == now.date())
    if is_expiry_today:
        st.success("🎯 **AAJ EXPIRY HAI!** — Real-time similarity analysis active!")
    elif expiry_dt:
        st.info(f"Next expiry: **{expiry_dt.strftime('%d %b %Y (%A)')}**")

    # ── Upload ─────────────────────────────────────────────────────────────────
    with st.expander("📂 Upload Historical NIFTY Daily Data (CSV)",
                     expanded="expiry_hist_df" not in st.session_state):
        st.markdown("""
**Required columns (any order, case-insensitive):** `Date, Open, High, Low, Close`
- Date formats accepted: `YYYY-MM-DD`, `DD-MM-YYYY`, `DD/MM/YYYY`, `DD-Mon-YYYY`
- Download from NSE Bhavcopy, Kite export, TradingView, or any broker
        """)
        uploaded = st.file_uploader("Choose CSV file", type=["csv"],
                                    key=f"expiry_hist_upload_{instrument}")
        if uploaded is not None:
            try:
                raw = pd.read_csv(uploaded)
                raw.columns = [c.strip() for c in raw.columns]
                # Flexible column mapping
                col_map = {}
                for c in raw.columns:
                    lc = c.lower().replace(" ", "").replace("_", "")
                    if "date" in lc:
                        col_map[c] = "date"
                    elif lc.startswith("open"):
                        col_map[c] = "open"
                    elif lc.startswith("high"):
                        col_map[c] = "high"
                    elif lc.startswith("low"):
                        col_map[c] = "low"
                    elif lc.startswith("close") or lc in ("ltp", "lastprice"):
                        col_map[c] = "close"
                    elif "vol" in lc:
                        col_map[c] = "volume"
                raw = raw.rename(columns=col_map)
                missing = [r for r in ["date", "open", "high", "low", "close"]
                           if r not in raw.columns]
                if missing:
                    st.error(f"Missing columns: {missing}. Got: {list(raw.columns)}")
                else:
                    raw["date"] = pd.to_datetime(raw["date"], dayfirst=True, errors="coerce")
                    raw.dropna(subset=["date"], inplace=True)
                    raw.sort_values("date", inplace=True)
                    for c in ["open", "high", "low", "close"]:
                        raw[c] = pd.to_numeric(
                            raw[c].astype(str).str.replace(",", ""), errors="coerce")
                    raw.dropna(subset=["open", "close"], inplace=True)
                    raw.reset_index(drop=True, inplace=True)
                    st.session_state["expiry_hist_df"] = raw
                    st.success(f"✅ Loaded **{len(raw):,}** rows | "
                               f"{raw['date'].min().strftime('%d %b %Y')} → "
                               f"{raw['date'].max().strftime('%d %b %Y')}")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")

    # ── Part 1: live intraday 5-min expiry-day charts + export (no CSV needed)
    render_intraday_expiry_section(expiry_dt, instrument=instrument)

    # ── Part 2: multi-year daily analysis & prediction (needs uploaded CSV) ──
    st.markdown("#### 📚 Multi-Year Historical Expiry Analysis & Prediction")
    hist_df = st.session_state.get("expiry_hist_df")
    if hist_df is None or hist_df.empty:
        st.info("⬆️ Upload NIFTY historical daily data above to see the multi-year "
                "expiry analysis and prediction.")
        return

    # ── Expiry-day weekday timeline (when did the expiry day change?) ─────────
    with st.expander("🗓️ NSE NIFTY Expiry-Day Rule Timeline (when it changed)",
                     expanded=False):
        st.dataframe(get_expiry_timeline_summary(), use_container_width=True, hide_index=True)
        st.caption("Detection below uses Thursday before 01-Sep-2025 and Tuesday after, "
                   "with automatic roll-back to the previous trading day on holidays.")

    # ── Identify expiry days using the historical weekday rule ────────────────
    # (Thursday pre-Sep-2025, Tuesday after; holiday-adjusted within each week)
    hist_df = hist_df.copy()
    hist_df["weekday"] = hist_df["date"].dt.weekday
    exp_mask = flag_expiry_days(hist_df["date"]).values
    exp_days = hist_df[exp_mask].copy()
    if exp_days.empty:
        st.warning("No expiry days detected in uploaded data — check the date column.")
        return

    # ── Per-expiry metrics ─────────────────────────────────────────────────────
    exp_days["chg_pct"] = ((exp_days["close"] - exp_days["open"])
                           / exp_days["open"] * 100).round(2)
    exp_days["range_pct"] = ((exp_days["high"] - exp_days["low"])
                             / exp_days["open"] * 100).round(2)
    exp_days["direction"] = exp_days["chg_pct"].apply(
        lambda x: "BULLISH" if x > 0.1 else "BEARISH" if x < -0.1 else "NEUTRAL")

    # Pre-expiry week: 5-day return into the expiry open
    pre_rets = []
    for _, erow in exp_days.iterrows():
        before = hist_df[hist_df["date"] < erow["date"]].tail(5)
        if len(before) >= 2:
            pw = (erow["open"] - float(before.iloc[0]["open"])) / float(before.iloc[0]["open"]) * 100
        else:
            pw = 0.0
        pre_rets.append(round(pw, 2))
    exp_days["pre_week_ret"] = pre_rets
    exp_days["pre_week_dir"] = exp_days["pre_week_ret"].apply(
        lambda x: "UP" if x > 0 else "DOWN")

    # ── Historical summary cards ───────────────────────────────────────────────
    st.markdown("#### 📊 Historical Expiry Day Summary")
    total = len(exp_days)
    bull = (exp_days["direction"] == "BULLISH").sum()
    bear = (exp_days["direction"] == "BEARISH").sum()
    neu = total - bull - bear
    avg_chg = exp_days["chg_pct"].mean()
    median_chg = exp_days["chg_pct"].median()
    avg_rng = exp_days["range_pct"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    chg_col = "#00ff88" if avg_chg >= 0 else "#ff4444"
    cards = [
        ("Expiry Days", f"{total}", f"{exp_days['date'].min().strftime('%b %Y')} – {exp_days['date'].max().strftime('%b %Y')}"),
        ("Bullish", f"{bull} ({bull/total*100:.0f}%)", "Close > Open", "#00ff88"),
        ("Bearish", f"{bear} ({bear/total*100:.0f}%)", "Close < Open", "#ff4444"),
        ("Avg Move", f"{avg_chg:+.2f}%", f"Median {median_chg:+.2f}%", chg_col),
        ("Avg Range", f"{avg_rng:.2f}%", "High–Low / Open", "#ffd700"),
    ]
    for col, (label, val, sub, *color) in zip([c1, c2, c3, c4, c5], cards):
        vc = color[0] if color else "#fff"
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="color:{vc};">{val}</div>
                <div class="metric-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    # ── Pre-week direction → expiry outcome table ──────────────────────────────
    st.markdown("#### 🔁 Pre-Expiry Week Trend → Expiry Day Outcome")
    for pw in ["UP", "DOWN"]:
        sub = exp_days[exp_days["pre_week_dir"] == pw]
        if sub.empty:
            continue
        s_bull = (sub["direction"] == "BULLISH").sum()
        s_bear = (sub["direction"] == "BEARISH").sum()
        n = len(sub)
        avg_mv = sub["chg_pct"].mean()
        bc = "#00ff88" if avg_mv >= 0 else "#ff4444"
        icon = "📈" if pw == "UP" else "📉"
        st.markdown(f"""
        <div style="background:#1e2130;border-radius:8px;padding:8px 14px;margin-bottom:6px;">
            {icon} <b>Pre-week {pw}</b> ({n} cases) &nbsp;→&nbsp;
            <span style="color:#00ff88;">Bullish {s_bull/n*100:.0f}%</span> &nbsp;|&nbsp;
            <span style="color:#ff4444;">Bearish {s_bear/n*100:.0f}%</span> &nbsp;|&nbsp;
            Avg move: <b style="color:{bc};">{avg_mv:+.2f}%</b>
        </div>""", unsafe_allow_html=True)

    # ── Visualisation: expiry day moves bar chart ──────────────────────────────
    st.markdown("#### 📈 Expiry Day Returns (Last 52 weeks)")
    recent_exp = exp_days.tail(52)
    bar_colors = ["#00cc66" if v >= 0 else "#cc2222" for v in recent_exp["chg_pct"]]
    fig_bar = go.Figure(go.Bar(
        x=recent_exp["date"].dt.strftime("%d%b'%y"),
        y=recent_exp["chg_pct"],
        marker_color=bar_colors,
        text=[f"{v:+.2f}%" for v in recent_exp["chg_pct"]],
        textposition="outside",
        textfont_size=9,
    ))
    fig_bar.update_layout(
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#ccc"), height=300,
        margin=dict(l=10, r=10, t=20, b=60),
        yaxis=dict(gridcolor="#1e2130", zeroline=True, zerolinecolor="#555"),
        xaxis=dict(tickangle=-60, gridcolor="#1e2130"),
        showlegend=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Calculate current pre-week return ─────────────────────────────────────
    # Prefer hist_df if it has data close to today; else use live candle data
    today_dt = now.date()
    recent_hist = hist_df[hist_df["date"].dt.date <= today_dt].tail(6)
    if len(recent_hist) >= 2 and (today_dt - recent_hist.iloc[-1]["date"].date()).days <= 5:
        curr_pw_ret = ((recent_hist.iloc[-1]["close"] - recent_hist.iloc[0]["open"])
                       / recent_hist.iloc[0]["open"] * 100)
    elif wide_candle_df is not None and not wide_candle_df.empty and spot > 0:
        w5 = wide_candle_df.head(5)
        curr_pw_ret = ((spot - float(w5.iloc[0]["open"])) / float(w5.iloc[0]["open"]) * 100
                       if not w5.empty else 0.0)
    else:
        curr_pw_ret = 0.0
    curr_pw_ret = round(curr_pw_ret, 2)
    curr_pw_dir = "UP" if curr_pw_ret > 0 else "DOWN"

    # ── Find similar historical expiry weeks ───────────────────────────────────
    # Similarity metric: same pre-week direction + closest pre-week return magnitude
    same_dir = exp_days[exp_days["pre_week_dir"] == curr_pw_dir].copy()
    pool = same_dir if len(same_dir) >= 5 else exp_days.copy()
    pool["sim_score"] = 1 / (1 + abs(pool["pre_week_ret"] - curr_pw_ret))
    top8 = pool.nlargest(8, "sim_score")

    pred_bull = (top8["direction"] == "BULLISH").sum()
    pred_bear = (top8["direction"] == "BEARISH").sum()
    pred_n = len(top8)
    pred_avg = top8["chg_pct"].mean()
    bull_pct = pred_bull / pred_n * 100 if pred_n else 50

    if bull_pct > 60:
        pred, pred_col, pred_icon = "BULLISH", "#00ff88", "📈"
        pred_txt = f"{bull_pct:.0f}% of the {pred_n} most similar expiry weeks closed green"
    elif bull_pct < 40:
        pred, pred_col, pred_icon = "BEARISH", "#ff4444", "📉"
        pred_txt = f"{100-bull_pct:.0f}% of the {pred_n} most similar expiry weeks closed red"
    else:
        pred, pred_col, pred_icon = "NEUTRAL", "#ffd700", "↔️"
        pred_txt = f"Mixed signals from {pred_n} similar historical expiry weeks"

    expiry_label = (f"🎯 TODAY'S EXPIRY — {expiry_dt.strftime('%d %b %Y')}"
                    if is_expiry_today
                    else f"📅 NEXT EXPIRY — {expiry_dt.strftime('%d %b %Y') if expiry_dt else '---'}")

    st.markdown("#### 🎯 Current Expiry Prediction")
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#15201a,#1e2130);
                border:2px solid {pred_col};border-radius:12px;
                padding:20px 24px;margin-bottom:16px;">
        <div style="font-size:12px;color:#888;margin-bottom:4px;">{expiry_label}</div>
        <div style="font-size:28px;font-weight:bold;color:{pred_col};">
            {pred_icon} {pred}
        </div>
        <div style="font-size:13px;color:#aaa;margin-top:2px;">{pred_txt}</div>
        <hr style="border-color:#2d3250;margin:12px 0;">
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;">
            <div>
                <div style="color:#888;font-size:11px;">Current pre-week</div>
                <div style="color:#fff;font-size:18px;font-weight:bold;">{curr_pw_ret:+.2f}%</div>
                <div style="font-size:11px;color:#888;">{curr_pw_dir}</div>
            </div>
            <div>
                <div style="color:#888;font-size:11px;">Predicted avg move</div>
                <div style="color:{pred_col};font-size:18px;font-weight:bold;">{pred_avg:+.2f}%</div>
                <div style="font-size:11px;color:#888;">Open → Close</div>
            </div>
            <div>
                <div style="color:#888;font-size:11px;">Bull / Bear split</div>
                <div style="font-size:18px;font-weight:bold;">
                    <span style="color:#00ff88;">{pred_bull}</span>
                    <span style="color:#888;"> / </span>
                    <span style="color:#ff4444;">{pred_bear}</span>
                </div>
                <div style="font-size:11px;color:#888;">of {pred_n} similar weeks</div>
            </div>
            <div>
                <div style="color:#888;font-size:11px;">Spot price</div>
                <div style="color:#fff;font-size:18px;font-weight:bold;">{spot:,.2f}</div>
                <div style="font-size:11px;color:#888;">NIFTY 50</div>
            </div>
        </div>
        <div style="margin-top:12px;font-size:11px;color:#555;">
            ⚠️ Based on historical pattern similarity only. Not financial advice.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Most similar historical weeks table ───────────────────────────────────
    st.markdown("#### 🔍 Most Similar Historical Expiry Days")
    disp = top8[["date", "open", "high", "low", "close", "chg_pct",
                 "range_pct", "direction", "pre_week_ret"]].copy()
    disp["date"] = disp["date"].dt.strftime("%d %b %Y")
    disp = disp.rename(columns={
        "date": "Date", "open": "Open", "high": "High", "low": "Low", "close": "Close",
        "chg_pct": "Chg %", "range_pct": "Range %",
        "direction": "Direction", "pre_week_ret": "Pre-week %",
    })
    num_fmt = {c: "{:.2f}" for c in ["Open", "High", "Low", "Close",
                                      "Chg %", "Range %", "Pre-week %"]}

    def style_dir_exp(val):
        if val == "BULLISH":
            return "background:#1a3a1a;color:#00ff88;font-weight:bold"
        if val == "BEARISH":
            return "background:#3a1a1a;color:#ff4444;font-weight:bold"
        return "color:#ffd700"

    styled_disp = style_cells(disp.style.format(num_fmt), style_dir_exp, ["Direction"])
    st.dataframe(styled_disp, use_container_width=True, hide_index=True)

    # ── Full history (collapsible) ─────────────────────────────────────────────
    with st.expander(f"📋 Full Expiry Day History ({total} days)"):
        all_disp = exp_days[["date", "open", "close", "chg_pct",
                              "range_pct", "direction", "pre_week_ret"]].copy()
        all_disp["date"] = all_disp["date"].dt.strftime("%d %b %Y")
        all_disp = all_disp.rename(columns={
            "date": "Date", "open": "Open", "close": "Close",
            "chg_pct": "Chg %", "range_pct": "Range %",
            "direction": "Direction", "pre_week_ret": "Pre-week %",
        })
        fmt2 = {c: "{:.2f}" for c in ["Open", "Close", "Chg %", "Range %", "Pre-week %"]}
        st2 = style_cells(
            all_disp.sort_values("Date", ascending=False).style.format(fmt2),
            style_dir_exp, ["Direction"],
        )
        st.dataframe(st2, use_container_width=True, hide_index=True)

    if st.button("🗑️ Clear uploaded data", key=f"clear_hist_data_{instrument}"):
        st.session_state.pop("expiry_hist_df", None)
        st.rerun()


def render_best_trade_tab(patterns, options_df: pd.DataFrame, spot: float, oi_delta: dict, greeks: dict):
    st.markdown("### 🏆 Best Trade Recommendation")
    market_open = is_market_open()

    if not market_open:
        st.error("🔴 Market Closed – No trade recommendations available.")
        return

    if not patterns:
        st.info("Scanning for high-probability setups...")
        return

    # Score patterns
    scored = []
    for pat in patterns:
        confidence = getattr(pat, "confidence", 0.5)
        if isinstance(confidence, str):
            conf_score = {"HIGH": 1.0, "MEDIUM": 0.6, "LOW": 0.3}.get(confidence, 0.5)
        else:
            conf_score = float(confidence)

        rr = float(pat.risk_reward) if pat.risk_reward else 0
        score = conf_score * 0.5 + min(rr / 5, 1.0) * 0.5

        # OI confirmation
        oi_bias = oi_delta.get("bias", "NEUTRAL")
        if (pat.signal == "BUY" and oi_bias == "BULLISH") or \
           (pat.signal == "SELL" and oi_bias == "BEARISH"):
            score += 0.2

        # Greeks confirmation
        g_bias = greeks.get("bias", "NEUTRAL")
        if (pat.signal == "BUY" and g_bias == "BULLISH") or \
           (pat.signal == "SELL" and g_bias == "BEARISH"):
            score += 0.15

        scored.append((score, pat))

    if not scored:
        st.info("No high-confidence setup found right now.")
        return

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_pat = scored[0]
    pat_name = getattr(best_pat, "pattern", getattr(best_pat, "name", "Signal"))
    confidence = getattr(best_pat, "confidence", "MEDIUM")
    if isinstance(confidence, (int, float)):
        confidence = "HIGH" if confidence > 0.7 else "MEDIUM" if confidence > 0.4 else "LOW"

    sig_color = "#00ff88" if best_pat.signal == "BUY" else "#ff4444"
    atm = round(spot / 50) * 50
    option_type = "CE" if best_pat.signal == "BUY" else "PE"
    oi_bias = oi_delta.get("bias", "NEUTRAL")
    g_bias = greeks.get("bias", "NEUTRAL")
    oi_confirm = "✅" if (best_pat.signal == "BUY" and oi_bias == "BULLISH") or \
                         (best_pat.signal == "SELL" and oi_bias == "BEARISH") else "⚠️"
    greeks_confirm = "✅" if (best_pat.signal == "BUY" and g_bias == "BULLISH") or \
                             (best_pat.signal == "SELL" and g_bias == "BEARISH") else "⚠️"

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#1a2a1a,#1e2130);border:2px solid {sig_color};
                border-radius:12px;padding:20px 24px;margin-bottom:16px;">
        <div style="font-size:24px;font-weight:bold;color:{sig_color};">
            {best_pat.signal} · {pat_name}
        </div>
        <div style="color:#aaa;margin:4px 0;">Confidence: <b>{confidence}</b> &nbsp;|&nbsp;
            Score: <b style="color:#ffd700;">{best_score:.2f}</b></div>
        <hr style="border-color:#333;margin:10px 0;">
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:12px;">
            <div><div style="color:#888;font-size:11px;">Entry</div>
                 <div style="font-size:20px;font-weight:bold;color:#fff;">{best_pat.entry:.2f}</div></div>
            <div><div style="color:#888;font-size:11px;">Stop Loss</div>
                 <div style="font-size:20px;font-weight:bold;color:#ff8888;">{best_pat.stop_loss:.2f}</div></div>
            <div><div style="color:#888;font-size:11px;">Target</div>
                 <div style="font-size:20px;font-weight:bold;color:#88ff88;">{best_pat.target:.2f}</div></div>
            <div><div style="color:#888;font-size:11px;">Risk:Reward</div>
                 <div style="font-size:20px;font-weight:bold;color:#ffd700;">1:{best_pat.risk_reward}</div></div>
        </div>
        <hr style="border-color:#333;margin:10px 0;">
        <div style="font-size:13px;color:#aaa;">
            Suggested Option: <b style="color:#fff;">NIFTY {atm} {option_type}</b> (ATM)
            &nbsp;|&nbsp; OI Confirm: {oi_confirm} {oi_bias}
            &nbsp;|&nbsp; Greeks: {greeks_confirm} {g_bias}
        </div>
        <div style="font-size:12px;color:#666;margin-top:6px;">
            {getattr(best_pat,'description','')}
        </div>
    </div>
    """, unsafe_allow_html=True)

    if len(scored) > 1:
        st.markdown("#### Other Setups")
        for score, pat in scored[1:4]:
            pn = getattr(pat, "pattern", getattr(pat, "name", "Signal"))
            sc = "#00ff88" if pat.signal == "BUY" else "#ff4444"
            st.markdown(f"""
            <div style="background:#1e2130;border-left:3px solid {sc};padding:8px 12px;
                        border-radius:0 6px 6px 0;margin-bottom:6px;">
                <span style="color:{sc};font-weight:bold;">{pat.signal}</span> · {pn}
                &nbsp;&nbsp;<span style="color:#888;font-size:12px;">Score: {score:.2f}</span>
                &nbsp;|&nbsp;Entry: {float(pat.entry):.2f} &nbsp;|&nbsp;SL: {float(pat.stop_loss):.2f}
                &nbsp;|&nbsp;T: {float(pat.target):.2f} &nbsp;|&nbsp;R:R 1:{pat.risk_reward}
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────
def render_connect_panel(connected: bool):
    """Show a Connect button + diagnostics when not connected to AngelOne."""
    if connected:
        return
    cols = st.columns([2, 6])
    with cols[0]:
        if st.button("🔌 Connect to AngelOne", type="primary", use_container_width=True):
            with st.spinner("Logging in to AngelOne…"):
                obj = get_client(force=True)
            if obj is not None:
                st.success("Connected! Loading live data…")
                st.rerun()
            else:
                st.error("Connection failed — see details below.")
    with cols[1]:
        err = get_last_error()
        if err:
            st.error(f"⚠️ {err}")
        else:
            st.info("Add your AngelOne API secrets, then click Connect.")
    with st.expander("ℹ️ Secrets format (Streamlit → Settings → Secrets)"):
        st.code(
            '[angel_one]\n'
            'api_key     = "your_api_key"\n'
            'client_id   = "your_client_code"\n'
            'mpin        = "your_mpin"        # or password = "..."\n'
            'totp_secret = "your_totp_base32_secret"\n',
            language="toml",
        )
    st.markdown("<hr style='border-color:#2d3250;margin:6px 0;'>", unsafe_allow_html=True)


def _render_instrument_section(instrument: str, selected_tf: int):
    """Fetch data and render all 7 sub-tabs for one instrument."""
    cfg = INSTRUMENT_CONFIG.get(instrument, INSTRUMENT_CONFIG["NIFTY"])

    # ── Fetch all data; never let one instrument's failure crash the page ─────
    try:
        spot = fetch_ltp_for(instrument)
        if spot and spot > 0:
            st.session_state[f"_last_ltp_{instrument}"] = spot
        else:
            spot = st.session_state.get(f"_last_ltp_{instrument}", 0.0)

        candle_df_wide = fetch_candle_data_for(instrument, selected_tf, 200)
        candle_df = filter_to_latest_day(candle_df_wide)

        candle_data_by_tf = {tf: fetch_candle_data_for(instrument, tf, 80)
                             for tf in [1, 2, 5, 10, 15, 30, 60]}

        expiry_dt = get_next_expiry_for(instrument)
        expiry_str = get_expiry_string(expiry_dt)
        options_df = fetch_options_chain_for(instrument, expiry_str)
    except Exception as e:
        st.error(f"⚠️ Could not load {cfg['display_name']} data: {e}")
        if not is_connected():
            st.info("🔌 Connect to AngelOne (top of page) to load live data.")
        return

    if options_df is None or options_df.empty:
        cached = st.session_state.get(f"_last_options_df_{instrument}")
        if cached is not None and not cached.empty:
            options_df = cached
    if options_df is not None and not options_df.empty:
        st.session_state[f"_last_options_df_{instrument}"] = options_df

    # If there's no price/candle data at all, show a clear message and stop.
    if (spot is None or spot <= 0) and (candle_df is None or candle_df.empty):
        exp_lbl = "monthly" if cfg.get("expiry_type") == "monthly" else "weekly"
        st.warning(
            f"📭 No live {cfg['display_name']} data available right now "
            f"(expiry cycle: {exp_lbl}). This can happen when the market is "
            f"closed or AngelOne hasn't returned data for this symbol/exchange "
            f"({cfg['exchange']}/{cfg['opt_exchange']})."
        )
        if not is_connected():
            st.info("🔌 Connect to AngelOne (top of page) to load live data.")
        return

    # Pattern detection
    patterns = []
    if candle_df is not None and not candle_df.empty:
        try:
            patterns = detect_all_patterns(candle_df)
        except Exception as e:
            st.warning(f"Pattern detection ({instrument}): {e}")

    # OI / Greeks analysis
    oi_delta, oi_annotations, greeks_result = {}, [], {}
    if options_df is not None and not options_df.empty:
        try:
            oi_delta = compute_delta_oi(options_df, spot)
            oi_annotations = get_oi_arrow_annotations(candle_df, options_df, spot, selected_tf)
        except Exception:
            pass
        try:
            greeks_result = analyze_greeks(options_df, spot)
        except Exception:
            pass

    # Expiry countdown banner
    countdown = get_expiry_countdown(expiry_dt)
    st.markdown(
        f"<div style='background:#1e2130;border-radius:6px;padding:6px 14px;"
        f"font-size:13px;color:#aaa;margin-bottom:8px;'>"
        f"<b style='color:#fff;'>{cfg['display_name']}</b> &nbsp;|&nbsp; "
        f"Spot: <b style='color:#00ff88;'>{spot:,.2f}</b> &nbsp;|&nbsp; "
        f"Expiry: <b style='color:#ffd700;'>{expiry_str}</b> &nbsp;|&nbsp; "
        f"<span style='color:#aaa;'>{countdown}</span></div>",
        unsafe_allow_html=True,
    )

    # Chart
    fig = build_chart(candle_df, patterns, oi_annotations, selected_tf,
                      instrument_label=cfg["display_name"])
    st.plotly_chart(fig, use_container_width=True, config={
        "displayModeBar": True,
        "displaylogo": False,
        "modeBarButtonsToRemove": ["pan2d", "lasso2d"],
    })

    # Sub-tabs
    t1, t2, t3, t4, t5, t6, t7 = st.tabs([
        "📋 Recommendations",
        "🎯 Strike Volume",
        "📝 Paper Trade",
        "📊 OI Table",
        "🔢 Greeks",
        "🏆 Best Trade",
        "📅 Expiry Analysis",
    ])
    with t1:
        render_recommendations_tab(patterns, spot, instrument=instrument)
    with t2:
        render_strike_volume_tab(options_df, spot, candle_data_by_tf, instrument=instrument)
    with t3:
        render_paper_trade_tab(patterns, spot, options_df=options_df,
                               candle_df=candle_df, instrument=instrument)
    with t4:
        render_oi_table_tab(candle_data_by_tf, options_df, spot)
    with t5:
        render_greeks_tab(options_df, spot)
    with t6:
        render_best_trade_tab(patterns, options_df, spot, oi_delta, greeks_result)
    with t7:
        render_expiry_analysis_tab(candle_df_wide, spot, expiry_dt, instrument=instrument)


def main():
    if not MODULES_OK:
        st.stop()

    # Fetch NIFTY LTP for header display
    ltp = fetch_ltp()
    if ltp and ltp > 0:
        st.session_state["_last_ltp"] = ltp
    else:
        ltp = st.session_state.get("_last_ltp", 0.0)
    spot_prev = ltp * 0.9985  # approximation for prev close display
    connected = is_connected()

    render_header(ltp, spot_prev, connected)
    render_connect_panel(connected)

    # Show scrip master / expiry diagnostic when data is unavailable
    if connected:
        diag = get_options_diagnostic()
        if diag:
            with st.expander("⚠️ Options data diagnostic (click to see why expiry/OI tabs show unavailable)", expanded=False):
                st.warning(diag)
                if st.button("🔄 Retry loading scrip master", key="retry_scrip"):
                    from modules.angelone_client import _MASTER_CACHE
                    _MASTER_CACHE["ts"] = 0.0  # force re-download on next call
                    st.cache_data.clear()
                    st.rerun()

    # Timeframe selector (shared across instruments)
    tf_options = {1: "1 min", 2: "2 min", 5: "5 min", 10: "10 min",
                  15: "15 min", 30: "30 min", 60: "60 min"}
    tf_col1, _ = st.columns([3, 9])
    with tf_col1:
        selected_tf = st.radio(
            "Chart Timeframe",
            options=list(tf_options.keys()),
            format_func=lambda x: tf_options[x],
            index=2,
            horizontal=True,
            key="chart_tf_radio",
        )

    # ── Instrument tabs ───────────────────────────────────────────────────────
    inst_tab_nifty, inst_tab_sensex, inst_tab_sbin = st.tabs([
        "📈 Nifty 50", "📊 Sensex", "🏦 SBIN"
    ])

    for _inst_tab, _inst_key in [
        (inst_tab_nifty, "NIFTY"),
        (inst_tab_sensex, "SENSEX"),
        (inst_tab_sbin, "SBIN"),
    ]:
        with _inst_tab:
            _render_instrument_section(_inst_key, selected_tf)

    # ── Auto-refresh bar at bottom ─────────────────────────────────────────
    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
    st.markdown("<hr style='border-color:#2d3250;margin:4px 0;'>", unsafe_allow_html=True)

    r1, r2, r3 = st.columns([5, 2, 1])
    with r1:
        market_msg = ("🟢 Live data streaming" if is_market_open()
                      else "🔴 Market closed – showing last session data")
        st.markdown(f"""
        <div class="refresh-bar">
            {market_msg} &nbsp;|&nbsp;
            Last update: <b>{datetime.now(IST).strftime('%H:%M:%S')}</b>
        </div>""", unsafe_allow_html=True)
    with r2:
        # Persist toggle in session state so it survives reruns
        if "autorefresh_on" not in st.session_state:
            st.session_state["autorefresh_on"] = False  # OFF by default
        toggled = st.toggle("Auto-refresh (5s)", value=st.session_state["autorefresh_on"],
                            key="ar_toggle")
        st.session_state["autorefresh_on"] = toggled
    with r3:
        if st.button("↺", help="Refresh now", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # Only fire the autorefresh component when toggled ON.
    # st_autorefresh triggers a full rerun which causes the grey flash; we place
    # it LAST and use debounce so only one rerun fires per interval.
    if st.session_state.get("autorefresh_on") and HAS_AUTOREFRESH:
        st_autorefresh(interval=5000, key="main_data_refresh", debounce=True)


if __name__ == "__main__":
    main()
