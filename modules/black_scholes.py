"""
Black-Scholes option pricing & Greeks for NIFTY index options.

The AngelOne `optionGreek` endpoint is not always reliable (often returns
zeros), so we compute Greeks locally from the live option LTP, spot, strike
and time-to-expiry. Implied volatility is solved via Newton-Raphson with a
bisection fallback for robustness — mirroring the approach in the reference
implementation.

Conventions returned (per-strike, per option type):
  • delta : CE in (0,1), PE in (-1,0)
  • gamma : same for CE/PE (1/point^2)
  • theta : per CALENDAR DAY (negative for long options)
  • vega  : per 1% change in IV
  • iv    : implied volatility in PERCENT (e.g. 12.5 == 12.5%)
"""

import math
import pandas as pd

# Default annualised risk-free rate (Indian T-bill ~6.5%)
RISK_FREE_RATE = 0.065
SECONDS_PER_YEAR = 365.0 * 24.0 * 3600.0


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _d1_d2(S: float, K: float, T: float, r: float, sigma: float):
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return None, None
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def bs_price(S: float, K: float, T: float, r: float, sigma: float, opt: str) -> float:
    """Black-Scholes theoretical price. opt = 'C' or 'P'."""
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    if d1 is None:
        # Degenerate: return intrinsic value
        if opt.upper().startswith("C"):
            return max(0.0, S - K)
        return max(0.0, K - S)
    disc = math.exp(-r * T)
    if opt.upper().startswith("C"):
        return S * _norm_cdf(d1) - K * disc * _norm_cdf(d2)
    return K * disc * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def _vega_raw(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """dPrice/dSigma for a 1.0 (100%) change in vol — used by the IV solver."""
    d1, _ = _d1_d2(S, K, T, r, sigma)
    if d1 is None:
        return 0.0
    return S * _norm_pdf(d1) * math.sqrt(T)


def implied_vol(price: float, S: float, K: float, T: float,
                r: float = RISK_FREE_RATE, opt: str = "C") -> float:
    """
    Solve for implied volatility (as a decimal, e.g. 0.12) from a market price.
    Newton-Raphson with a bisection fallback. Returns 0.0 if it cannot solve.
    """
    if price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return 0.0

    # Price cannot be below intrinsic — clamp to avoid non-convergence
    intrinsic = max(0.0, S - K) if opt.upper().startswith("C") else max(0.0, K - S)
    if price < intrinsic:
        price = intrinsic + 1e-4

    sigma = 0.25  # initial guess
    for _ in range(60):
        model = bs_price(S, K, T, r, sigma, opt)
        diff = model - price
        if abs(diff) < 1e-4:
            return max(sigma, 0.0)
        v = _vega_raw(S, K, T, r, sigma)
        if v < 1e-8:
            break
        sigma -= diff / v
        if sigma <= 1e-4:
            sigma = 1e-4
        elif sigma > 5.0:
            sigma = 5.0

    # Bisection fallback between 0.1% and 500% vol
    lo, hi = 1e-4, 5.0
    plo = bs_price(S, K, T, r, lo, opt) - price
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        pmid = bs_price(S, K, T, r, mid, opt) - price
        if abs(pmid) < 1e-4:
            return mid
        if (plo < 0) == (pmid < 0):
            lo, plo = mid, pmid
        else:
            hi = mid
    return max(0.5 * (lo + hi), 0.0)


def bs_greeks(S: float, K: float, T: float, r: float, sigma: float, opt: str) -> dict:
    """
    Return delta, gamma, theta (per calendar day), vega (per 1% IV).
    sigma is a decimal (0.12 == 12%).
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    if d1 is None:
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}

    pdf = _norm_pdf(d1)
    disc = math.exp(-r * T)
    sqrtT = math.sqrt(T)
    is_call = opt.upper().startswith("C")

    if is_call:
        delta = _norm_cdf(d1)
        theta_annual = (-(S * pdf * sigma) / (2.0 * sqrtT)
                        - r * K * disc * _norm_cdf(d2))
    else:
        delta = _norm_cdf(d1) - 1.0
        theta_annual = (-(S * pdf * sigma) / (2.0 * sqrtT)
                        + r * K * disc * _norm_cdf(-d2))

    gamma = pdf / (S * sigma * sqrtT)
    vega = S * pdf * sqrtT / 100.0          # per 1% change in IV
    theta = theta_annual / 365.0            # per calendar day

    return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega}


def time_to_expiry_years(expiry_dt, now_dt) -> float:
    """
    Years to expiry (calendar time) until 15:30 IST on the expiry day.
    Floors at a tiny positive value so Greeks stay finite on expiry day.
    """
    if expiry_dt is None:
        return 0.0
    try:
        close = expiry_dt.replace(hour=15, minute=30, second=0, microsecond=0)
        secs = (close - now_dt).total_seconds()
        if secs <= 0:
            secs = 3600.0  # ~1h floor after close so greeks don't blow up
        return secs / SECONDS_PER_YEAR
    except Exception:
        return 0.0


def compute_chain_greeks(df: pd.DataFrame, spot: float, expiry_dt, now_dt,
                         r: float = RISK_FREE_RATE) -> pd.DataFrame:
    """
    Fill ce_*/pe_* greek columns (delta, gamma, theta, vega, iv) on an option
    chain DataFrame using Black-Scholes, derived from ce_ltp / pe_ltp.
    IV is stored in PERCENT. Modifies a copy and returns it.
    """
    if df is None or df.empty or spot <= 0:
        return df

    T = time_to_expiry_years(expiry_dt, now_dt)
    out = df.copy()

    for side, opt in (("ce", "C"), ("pe", "P")):
        ivs, deltas, gammas, thetas, vegas = [], [], [], [], []
        for _, row in out.iterrows():
            K = float(row["strike"])
            price = float(row.get(f"{side}_ltp", 0) or 0)
            if price <= 0 or T <= 0:
                ivs.append(0.0); deltas.append(0.0); gammas.append(0.0)
                thetas.append(0.0); vegas.append(0.0)
                continue
            sigma = implied_vol(price, spot, K, T, r, opt)
            if sigma <= 0:
                ivs.append(0.0); deltas.append(0.0); gammas.append(0.0)
                thetas.append(0.0); vegas.append(0.0)
                continue
            g = bs_greeks(spot, K, T, r, sigma, opt)
            ivs.append(round(sigma * 100.0, 2))     # percent
            deltas.append(round(g["delta"], 4))
            gammas.append(round(g["gamma"], 6))
            thetas.append(round(g["theta"], 2))
            vegas.append(round(g["vega"], 2))
        out[f"{side}_iv"] = ivs
        out[f"{side}_delta"] = deltas
        out[f"{side}_gamma"] = gammas
        out[f"{side}_theta"] = thetas
        out[f"{side}_vega"] = vegas

    return out
