"""
ROUTER-PORTFOLIO-20260628
Sprint 5F — Task 3: Portfolio VaR endpoint
GET /api/portfolio?tickers=BBCA,BBRI&prices=9800,4500&lots=5,10&risk_pct=2

Multi-ticker portfolio VaR using:
- Per-ticker ATR-based risk (from risk_service.py)
- Historical Volatility 20d (hvol_20d ÷ 100 CRITICAL per risk_service convention)
- Portfolio-level VaR via covariance matrix (Variance-Covariance method)
- Circuit breaker: 15% max portfolio drawdown
"""
import os
import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel

log = logging.getLogger(__name__)

router = APIRouter(prefix='/api/portfolio', tags=['portfolio'])

SUFFIX       = '.JK'
TRADING_DAYS = 252
CONFIDENCE   = 0.95          # 95% VaR
Z_95         = 1.645
CIRCUIT_BREAK_PCT = 0.15     # 15% max drawdown gate


# ── Schemas ───────────────────────────────────────────────────────────────────

class PositionIn(BaseModel):
    ticker:     str
    price:      float         # current market price IDR
    lots:       int           # number of lots (1 lot = 100 shares)
    entry_price: Optional[float] = None   # for unrealized P&L

class PortfolioRequest(BaseModel):
    positions:     list[PositionIn]
    portfolio_idr: float = 100_000_000    # total portfolio value IDR
    peak_value:    Optional[float] = None # for drawdown calc
    confidence:    float = 0.95
    holding_days:  int   = 3              # IDX swing typical hold

class TickerRisk(BaseModel):
    ticker:          str
    market_value:    float
    position_pct:    float
    daily_vol_pct:   float     # hist_vol_20d (annualized ÷ sqrt(252) for daily)
    atr_14:          float
    atr_pct:         float     # atr / price
    position_var_1d: float     # single-position 1-day VaR IDR (95%)
    position_var_3d: float     # scaled to 3-day VaR
    unrealized_pnl:  Optional[float] = None

class PortfolioVaRResponse(BaseModel):
    success:          bool
    ts:               str
    portfolio_idr:    float
    invested_idr:     float
    cash_idr:         float
    positions:        list[TickerRisk]
    # Portfolio-level VaR
    portfolio_var_1d: float    # IDR, 95% confidence
    portfolio_var_3d: float
    portfolio_var_pct_1d: float
    portfolio_var_pct_3d: float
    # Diversification
    correlation_avg:  Optional[float]
    diversification_ratio: float   # undiversified / diversified VaR
    # Risk gates
    current_drawdown_pct: Optional[float]
    circuit_breaker_triggered: bool
    circuit_breaker_pct: float
    # Summary
    risk_level:       str      # LOW / MEDIUM / HIGH / CRITICAL


# ── Helpers ───────────────────────────────────────────────────────────────────

def fetch_returns(tickers: list[str], days: int = 60) -> pd.DataFrame:
    """Download adjusted close and compute daily returns. Returns DataFrame."""
    end   = datetime.now()
    start = end - timedelta(days=days + 10)
    yf_tickers = [t + SUFFIX for t in tickers]

    df = yf.download(yf_tickers, start=start, end=end,
                     interval='1d', progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        closes = df['Close']
    else:
        closes = df[['Close']]

    # Rename columns: BBCA.JK → BBCA
    closes.columns = [c.replace(SUFFIX, '') for c in closes.columns]
    returns = closes.pct_change().dropna()
    return returns


def fetch_single_atr(ticker: str) -> tuple[float, float, float]:
    """
    Returns (atr14, hist_vol_20d, close).
    hist_vol_20d is raw daily std * sqrt(252) — NOT ÷100 here (kept as decimal).
    """
    end   = datetime.now()
    start = end - timedelta(days=60)
    df = yf.download(ticker + SUFFIX, start=start, end=end,
                     interval='1d', progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.empty or len(df) < 15:
        return 0.0, 0.0, 0.0

    c = df['Close']
    h = df['High']
    l = df['Low']

    # ATR14
    tr   = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr  = float(tr.rolling(14).mean().iloc[-1])
    # hvol_20d — annualized, stored as decimal (e.g. 0.35 = 35%)
    # NOTE: hvol_20d ÷ 100 per risk_service convention means we divide by 100 only
    #       when it's stored as integer percent (e.g. 35 not 0.35).
    #       Here yfinance returns fractional returns → std is already decimal.
    hvol = float(c.pct_change().rolling(20).std().iloc[-1] * np.sqrt(TRADING_DAYS))
    close = float(c.iloc[-1])
    return atr, hvol, close


def risk_level_label(var_pct_3d: float) -> str:
    if var_pct_3d < 0.02:    return 'LOW'
    elif var_pct_3d < 0.04:  return 'MEDIUM'
    elif var_pct_3d < 0.08:  return 'HIGH'
    else:                     return 'CRITICAL'


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post('', response_model=PortfolioVaRResponse)
async def portfolio_var(req: PortfolioRequest):
    """
    Multi-ticker Portfolio VaR calculation.

    Method: Variance-Covariance (parametric), 95% confidence.
    VaR(3d) = VaR(1d) × sqrt(3)  [square-root-of-time scaling]
    """
    if not req.positions:
        raise HTTPException(status_code=400, detail='No positions provided')
    if len(req.positions) > 10:
        raise HTTPException(status_code=400, detail='Max 10 positions')

    tickers = [p.ticker for p in req.positions]
    confidence_z = Z_95 if abs(req.confidence - 0.95) < 0.001 else 2.326  # 99%

    # ── Fetch price data ──────────────────────────────────────────────────────
    try:
        returns_df = fetch_returns(tickers, days=60)
    except Exception as e:
        log.error(f"Price fetch error: {e}")
        raise HTTPException(status_code=502, detail=f'Price fetch failed: {e}')

    # ── Per-position risk ─────────────────────────────────────────────────────
    position_risks: list[TickerRisk] = []
    weights        = []
    daily_vols     = []
    invested_total = 0.0

    for pos in req.positions:
        ticker = pos.ticker
        shares = pos.lots * 100
        mkt_val = pos.price * shares
        invested_total += mkt_val

        # Fetch ATR + hvol from live data
        atr, hvol, live_close = fetch_single_atr(ticker)
        # Use request price if live fetch failed
        if live_close <= 0:
            live_close = pos.price

        # daily_vol = hvol / sqrt(252) for single-day vol
        daily_vol = hvol / np.sqrt(TRADING_DAYS)
        atr_pct   = atr / live_close if live_close > 0 else 0.0

        # Single-position parametric VaR
        var_1d = confidence_z * daily_vol * mkt_val
        var_3d = var_1d * np.sqrt(req.holding_days)

        unrealized = None
        if pos.entry_price and pos.entry_price > 0:
            unrealized = (pos.price - pos.entry_price) * shares

        position_risks.append(TickerRisk(
            ticker          = ticker,
            market_value    = mkt_val,
            position_pct    = 0.0,   # filled after total known
            daily_vol_pct   = round(daily_vol * 100, 3),
            atr_14          = round(atr, 2),
            atr_pct         = round(atr_pct * 100, 3),
            position_var_1d = round(var_1d, 0),
            position_var_3d = round(var_3d, 0),
            unrealized_pnl  = round(unrealized, 0) if unrealized is not None else None,
        ))
        weights.append(mkt_val)
        daily_vols.append(daily_vol)

    # Update position_pct
    if invested_total > 0:
        for risk in position_risks:
            risk.position_pct = round(risk.market_value / invested_total * 100, 2)

    # ── Portfolio VaR (covariance matrix) ────────────────────────────────────
    # Build weight vector (fraction of invested)
    w = np.array(weights) / invested_total if invested_total > 0 else np.ones(len(weights))
    sigma = np.array(daily_vols)

    # Correlation matrix from returns
    available = [t for t in tickers if t in returns_df.columns]
    corr_matrix = np.eye(len(tickers))
    corr_avg = None

    if len(available) >= 2:
        try:
            corr_df = returns_df[available].corr()
            # Fill in corr for available tickers
            for i, ti in enumerate(tickers):
                for j, tj in enumerate(tickers):
                    if ti in corr_df.index and tj in corr_df.columns:
                        corr_matrix[i, j] = corr_df.loc[ti, tj]
            off_diag = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
            corr_avg = round(float(off_diag.mean()), 3) if len(off_diag) > 0 else None
        except Exception as e:
            log.warning(f"Correlation calc failed: {e}")

    # Covariance matrix: Sigma_ij = corr_ij * sigma_i * sigma_j
    cov_matrix = np.outer(sigma, sigma) * corr_matrix

    # Portfolio variance = w^T * Cov * w
    port_var_1d_variance = float(w @ cov_matrix @ w)
    port_vol_1d = np.sqrt(port_var_1d_variance)

    port_var_1d_idr = confidence_z * port_vol_1d * invested_total
    port_var_3d_idr = port_var_1d_idr * np.sqrt(req.holding_days)

    # Undiversified VaR (sum of individual VaRs)
    undiv_var_1d = sum(r.position_var_1d for r in position_risks)
    div_ratio = round(undiv_var_1d / port_var_1d_idr, 3) if port_var_1d_idr > 0 else 1.0

    # ── Circuit breaker ───────────────────────────────────────────────────────
    peak = req.peak_value or req.portfolio_idr
    current_dd = None
    cb_triggered = False
    if peak > 0:
        current_dd = round((peak - req.portfolio_idr) / peak * 100, 2)
        cb_triggered = current_dd >= CIRCUIT_BREAK_PCT * 100

    cash_idr = req.portfolio_idr - invested_total

    return PortfolioVaRResponse(
        success          = True,
        ts               = datetime.utcnow().isoformat(),
        portfolio_idr    = req.portfolio_idr,
        invested_idr     = round(invested_total, 0),
        cash_idr         = round(cash_idr, 0),
        positions        = position_risks,
        portfolio_var_1d = round(port_var_1d_idr, 0),
        portfolio_var_3d = round(port_var_3d_idr, 0),
        portfolio_var_pct_1d = round(port_var_1d_idr / req.portfolio_idr * 100, 3),
        portfolio_var_pct_3d = round(port_var_3d_idr / req.portfolio_idr * 100, 3),
        correlation_avg  = corr_avg,
        diversification_ratio = div_ratio,
        current_drawdown_pct  = current_dd,
        circuit_breaker_triggered = cb_triggered,
        circuit_breaker_pct = CIRCUIT_BREAK_PCT * 100,
        risk_level = risk_level_label(port_var_3d_idr / req.portfolio_idr),
    )


@router.get('/health')
async def portfolio_health():
    return {'status': 'ok', 'endpoint': '/api/portfolio'}
