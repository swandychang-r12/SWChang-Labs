"""
risk_service.py — Phase 5D: Risk model upgrade
- ATR-based position sizing (from XGBoost feature D01_atr_14 + D02_atr_ratio)
- Simplified Portfolio VaR (rho=0.5 IDX correlation assumption)
- Drawdown circuit breaker

ATR Position Sizing:
  risk_per_trade = portfolio_value * risk_pct  (default 1%)
  stop_distance  = ATR * atr_multiplier         (default 2.0x)
  shares         = risk_per_trade / stop_distance
  position_value = shares * price

Portfolio VaR (2-asset simplified, rho=0.5):
  sigma_daily from hvol_20d (historical volatility)
  VaR_1d = portfolio_value * z * sqrt(w1^2*s1^2 + w2^2*s2^2 + 2*rho*w1*s1*w2*s2)
  z = 1.645 (95% confidence)
"""
import math
import logging
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_PORTFOLIO_VALUE = 50_000_000   # IDR 50 juta default
DEFAULT_RISK_PCT = 0.01                # 1% risk per trade
ATR_MULTIPLIER = 2.0                   # 2x ATR as stop distance
VAR_Z_95 = 1.645                       # 95% confidence one-tail
VAR_Z_99 = 2.326                       # 99% confidence
IDX_RHO = 0.5                          # Simplified IDX avg correlation
MAX_DRAWDOWN_THRESHOLD = 0.15          # 15% max portfolio drawdown → circuit breaker
MAX_POSITION_PCT = 0.20                # Hard cap: 20% of portfolio per position


def get_ticker_features(ticker: str) -> Optional[dict]:
    """Load relevant risk features from features.parquet."""
    try:
        import pandas as pd
        scanner = Path("/trading-scanner")
        fp = scanner / "data" / "features.parquet"
        if not fp.exists():
            return None

        df = pd.read_parquet(fp)
        ticker_jk = ticker if ticker.endswith(".JK") else f"{ticker}.JK"

        if "ticker" in df.columns:
            mask = df["ticker"].isin([ticker, ticker_jk])
        else:
            mask = df.index.isin([ticker, ticker_jk])

        row = df[mask].tail(1)
        if row.empty:
            return None

        def g(col, default=0.0):
            return float(row[col].iloc[0]) if col in row.columns else default

        return {
            "atr_14":       g("D01_atr_14"),        # ATR in price points
            "atr_ratio":    g("D02_atr_ratio"),      # ATR / close price
            "hvol_20d":     g("D11_hvol_20d"),       # 20-day historical volatility (annualized)
            "bb_width":     g("D07_bb_width"),       # Bollinger band width (volatility proxy)
            "price_ret_1d": g("A06_price_return_1d"),
        }
    except Exception as e:
        logger.error(f"get_ticker_features error: {e}")
        return None


def compute_atr_position_sizing(
    ticker: str,
    current_price: float,
    portfolio_value: float = DEFAULT_PORTFOLIO_VALUE,
    risk_pct: float = DEFAULT_RISK_PCT,
    atr_multiplier: float = ATR_MULTIPLIER
) -> dict:
    """
    ATR-based position sizing.
    Returns: shares, position_value, stop_loss, take_profit, risk_amount
    """
    features = get_ticker_features(ticker)

    if features is None or features["atr_14"] <= 0:
        # Fallback: use 2% of price as ATR estimate
        atr = current_price * 0.02
        atr_ratio = 0.02
        source = "fallback_2pct"
    else:
        atr = features["atr_14"]
        atr_ratio = features["atr_ratio"]
        source = "xgboost_features"

    # Risk amount in IDR
    risk_amount = portfolio_value * risk_pct

    # Stop distance = ATR * multiplier (in price units)
    stop_distance = atr * atr_multiplier

    # Shares to buy so that losing stop_distance per share = risk_amount
    shares = risk_amount / stop_distance if stop_distance > 0 else 0
    shares = int(math.floor(shares / 100) * 100)  # IDX: round to lot (100 shares)

    # Position value
    position_value = shares * current_price

    # Hard cap: max 20% of portfolio
    if position_value > portfolio_value * MAX_POSITION_PCT:
        shares = int(math.floor((portfolio_value * MAX_POSITION_PCT / current_price) / 100) * 100)
        position_value = shares * current_price

    stop_loss = current_price - stop_distance
    take_profit = current_price + (stop_distance * 2.0)  # 2:1 R/R minimum

    return {
        "ticker": ticker,
        "current_price": current_price,
        "shares": shares,
        "lots": shares // 100,
        "position_value_idr": position_value,
        "position_pct": position_value / portfolio_value if portfolio_value > 0 else 0,
        "stop_loss": round(stop_loss, 0),
        "take_profit": round(take_profit, 0),
        "stop_distance_pct": stop_distance / current_price if current_price > 0 else 0,
        "risk_amount_idr": min(risk_amount, position_value - (shares * stop_loss) if shares > 0 else risk_amount),
        "atr_14": atr,
        "atr_ratio": atr_ratio,
        "atr_source": source,
        "rr_ratio": 2.0,
        "portfolio_value": portfolio_value,
        "risk_pct": risk_pct
    }


def compute_portfolio_var(
    positions: list,  # [{"ticker": str, "weight": float, "sigma_daily": float}]
    portfolio_value: float = DEFAULT_PORTFOLIO_VALUE,
    confidence: float = 0.95,
    horizon_days: int = 1
) -> dict:
    """
    Simplified Portfolio VaR with constant pairwise correlation IDX_RHO=0.5.
    Requires positions list with weights and daily sigma per stock.
    """
    if not positions:
        return {"var_idr": 0, "var_pct": 0, "method": "empty"}

    z = VAR_Z_95 if confidence == 0.95 else VAR_Z_99

    n = len(positions)
    # Portfolio variance: sum(w_i^2 * s_i^2) + rho * sum_{i!=j}(w_i*s_i*w_j*s_j)
    var_sum = 0.0
    cross_sum = 0.0

    for i, p in enumerate(positions):
        wi = p["weight"]
        si = p["sigma_daily"]
        var_sum += (wi * si) ** 2
        for j, q in enumerate(positions):
            if i != j:
                wj = q["weight"]
                sj = q["sigma_daily"]
                cross_sum += IDX_RHO * wi * si * wj * sj

    portfolio_sigma = math.sqrt(var_sum + cross_sum) * math.sqrt(horizon_days)
    var_pct = z * portfolio_sigma
    var_idr = portfolio_value * var_pct

    return {
        "var_idr": round(var_idr, 0),
        "var_pct": round(var_pct * 100, 3),  # as percentage
        "portfolio_sigma_daily": round(portfolio_sigma, 4),
        "confidence": confidence,
        "horizon_days": horizon_days,
        "n_positions": n,
        "rho_assumption": IDX_RHO,
        "method": "simplified_constant_corr"
    }


def compute_single_stock_var(
    ticker: str,
    position_value: float,
    confidence: float = 0.95,
    horizon_days: int = 1
) -> dict:
    """VaR for a single stock position using hvol_20d from features."""
    features = get_ticker_features(ticker)

    if features and features["hvol_20d"] > 0:
        hvol_annual = features["hvol_20d"]  # annualized
        sigma_daily = (hvol_annual / 100.0) / math.sqrt(252)
        source = "hvol_20d"
    else:
        # IDX average daily vol ~1.5%
        sigma_daily = 0.015
        source = "fallback_idx_avg"

    z = VAR_Z_95 if confidence == 0.95 else VAR_Z_99
    var_pct = z * sigma_daily * math.sqrt(horizon_days)
    var_idr = position_value * var_pct

    return {
        "ticker": ticker,
        "position_value": position_value,
        "var_1d_idr": round(var_idr, 0),
        "var_1d_pct": round(var_pct * 100, 3),
        "sigma_daily_pct": round(sigma_daily * 100, 3),
        "sigma_source": source,
        "confidence": confidence,
        "z_score": z
    }


def check_drawdown_circuit_breaker(
    current_portfolio_value: float,
    peak_portfolio_value: float,
    max_drawdown_threshold: float = MAX_DRAWDOWN_THRESHOLD
) -> dict:
    """
    Circuit breaker: if current drawdown > threshold → block new positions.
    Returns: {"breaker_open": bool, "drawdown_pct": float, "message": str}
    """
    if peak_portfolio_value <= 0:
        return {
            "breaker_open": False,
            "drawdown_pct": 0.0,
            "threshold_pct": max_drawdown_threshold * 100,
            "message": "No peak value — circuit breaker inactive"
        }

    drawdown = (peak_portfolio_value - current_portfolio_value) / peak_portfolio_value
    breaker_open = drawdown >= max_drawdown_threshold

    return {
        "breaker_open": breaker_open,
        "drawdown_pct": round(drawdown * 100, 2),
        "threshold_pct": round(max_drawdown_threshold * 100, 1),
        "current_value": current_portfolio_value,
        "peak_value": peak_portfolio_value,
        "drawdown_idr": round(peak_portfolio_value - current_portfolio_value, 0),
        "message": (
            f"⛔ CIRCUIT BREAKER OPEN: Drawdown {drawdown*100:.1f}% >= {max_drawdown_threshold*100:.0f}% threshold. "
            "Stop all new positions until recovered."
            if breaker_open else
            f"✅ OK: Drawdown {drawdown*100:.1f}% < {max_drawdown_threshold*100:.0f}% threshold"
        )
    }
