"""
data_quality_service.py — Phase 5E: Data quality + IDX edge
1. yfinance validation (staleness, zero-volume, price sanity)
2. ARA/ARB handler for IDX (Auto Rejection Above/Below)
3. Data health summary
"""
import logging
from datetime import datetime, date
from typing import Optional, Dict, Any

import pytz

logger = logging.getLogger(__name__)

WIB = pytz.timezone("Asia/Jakarta")

# IDX ARA/ARB rules (as of 2024+)
# Regular board (Papan Utama/Pengembangan): ±25%
# New listing (7 days): +35% / -10%
# Acceleration board (Papan Akselerasi): +35% / -10% (always)
# Penny stocks < 50 IDR: no limit
ARA_REGULAR_PCT = 25.0      # %
ARB_REGULAR_PCT = 25.0      # %
ARA_ACCELERATION_PCT = 35.0
ARB_ACCELERATION_PCT = 10.0

# Data staleness: if market closed within last 2 trading days, data should be fresh
DATA_STALE_DAYS = 2
# Price sanity bounds: yfinance occasionally returns wrong values
MIN_PRICE_IDR = 50.0
MAX_PRICE_IDR = 100_000_000.0
MAX_SINGLE_DAY_MOVE_PCT = 40.0  # If move > ARA+buffer → flag as suspicious


def validate_ohlcv(ohlcv: Optional[Dict], ticker: str) -> Dict:
    """
    Validate yfinance OHLCV data for common IDX data quality issues.
    Returns: {"valid": bool, "issues": [...], "warnings": [...]}
    """
    issues = []
    warnings = []

    if ohlcv is None:
        return {"valid": False, "issues": ["no_data: yfinance returned None"], "warnings": []}

    # 1. Staleness check
    try:
        data_date = datetime.strptime(ohlcv["date"], "%Y-%m-%d").date()
        today = datetime.now(WIB).date()
        days_old = (today - data_date).days

        # Skip if weekend (no trading)
        if today.weekday() < 5:  # weekday
            if days_old > DATA_STALE_DAYS:
                issues.append(f"stale_data: {days_old} days old (last: {ohlcv['date']})")
        elif days_old > DATA_STALE_DAYS + 2:  # weekend allowance
            warnings.append(f"possibly_stale: {days_old} days old (weekend)")
    except Exception:
        warnings.append("date_parse_error: could not parse date")

    # 2. Price sanity
    close = ohlcv.get("close", 0)
    if close < MIN_PRICE_IDR:
        issues.append(f"price_below_min: close={close} < {MIN_PRICE_IDR}")
    elif close > MAX_PRICE_IDR:
        issues.append(f"price_above_max: close={close} > {MAX_PRICE_IDR}")

    # 3. Zero volume (holiday, suspension, or bad data)
    volume = ohlcv.get("volume", 0)
    if volume == 0:
        issues.append("zero_volume: no trading activity — possible suspension or holiday")
    elif volume < 100:  # Less than 1 lot — highly unusual for OHLCV
        warnings.append(f"very_low_volume: {volume} shares")

    # 4. Price move sanity (OHLCV internal)
    high = ohlcv.get("high", 0)
    low = ohlcv.get("low", 0)
    open_price = ohlcv.get("open", 0)
    if open_price > 0 and high > 0 and low > 0:
        day_range_pct = (high - low) / open_price * 100
        if day_range_pct > MAX_SINGLE_DAY_MOVE_PCT:
            warnings.append(
                f"suspicious_range: H-L range {day_range_pct:.1f}% "
                f"(open={open_price}, high={high}, low={low}) — possible ARA/ARB or data error"
            )

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "data_date": ohlcv.get("date"),
        "close": close,
        "volume": volume
    }


def compute_ara_arb(
    current_price: float,
    prev_close: float,
    board: str = "regular",  # "regular" | "acceleration" | "new_listing"
    listing_date: Optional[date] = None
) -> Dict:
    """
    Compute ARA (Auto Rejection Above) and ARB (Auto Rejection Below) limits.
    IDX rules:
      Regular board: ±25%
      Acceleration/new listing (within 7 days): +35% / -10%
    """
    if prev_close <= 0:
        return {"error": "prev_close must be > 0"}

    # Determine board type
    if board == "acceleration":
        ara_pct = ARA_ACCELERATION_PCT
        arb_pct = ARB_ACCELERATION_PCT
    elif board == "new_listing" and listing_date:
        days_since_listing = (date.today() - listing_date).days
        if days_since_listing <= 7:
            ara_pct = ARA_ACCELERATION_PCT
            arb_pct = ARB_ACCELERATION_PCT
        else:
            ara_pct = ARA_REGULAR_PCT
            arb_pct = ARB_REGULAR_PCT
    else:
        ara_pct = ARA_REGULAR_PCT
        arb_pct = ARB_REGULAR_PCT

    # Compute limits (rounded to nearest 1 IDR)
    ara_price = round(prev_close * (1 + ara_pct / 100))
    arb_price = round(prev_close * (1 - arb_pct / 100))

    # Compute distance from current price
    if current_price > 0:
        to_ara_pct = (ara_price - current_price) / current_price * 100
        to_arb_pct = (arb_price - current_price) / current_price * 100
    else:
        to_ara_pct = ara_pct
        to_arb_pct = -arb_pct

    # Check if already at limit
    at_ara = current_price >= ara_price * 0.999  # within 0.1% of ARA
    at_arb = current_price <= arb_price * 1.001

    return {
        "prev_close": prev_close,
        "current_price": current_price,
        "board": board,
        "ara_price": ara_price,
        "arb_price": arb_price,
        "ara_pct": ara_pct,
        "arb_pct": arb_pct,
        "to_ara_pct": round(to_ara_pct, 2),
        "to_arb_pct": round(to_arb_pct, 2),
        "at_ara": at_ara,
        "at_arb": at_arb,
        "status": "AT_ARA" if at_ara else ("AT_ARB" if at_arb else "NORMAL"),
        "note": (
            "⛔ AT ARA — cannot buy higher. Risk of gap down at open tomorrow."
            if at_ara else
            "⛔ AT ARB — cannot sell lower. Risk of continued selling."
            if at_arb else
            "Normal trading range"
        )
    }


async def get_data_health(ticker: str, db_session=None) -> Dict:
    """
    Full data health check for a ticker:
    1. yfinance fetch + validation
    2. ARA/ARB limits
    3. Data freshness from Postgres OHLCV
    """
    from app.services.data_fetcher import fetch_ohlcv

    health = {
        "ticker": ticker,
        "checked_at": datetime.now(WIB).isoformat(),
        "yfinance": {},
        "ara_arb": {},
        "db_freshness": {}
    }

    # 1. Fetch + validate from yfinance
    try:
        ohlcv = await fetch_ohlcv(ticker, period="5d", interval="1d", db_session=db_session)
        validation = validate_ohlcv(ohlcv, ticker)
        health["yfinance"] = {
            "ohlcv": ohlcv,
            "validation": validation
        }
        current_price = ohlcv["close"] if ohlcv else 0

        # Need prev_close for ARA/ARB (fetch 2 bars, compare last two)
        import yfinance as yf
        ticker_jk = ticker if ticker.endswith(".JK") else f"{ticker}.JK"
        stock = yf.Ticker(ticker_jk)
        df = stock.history(period="5d", interval="1d")
        if len(df) >= 2:
            prev_close = float(df.iloc[-2]["Close"])
            cur_close = float(df.iloc[-1]["Close"])
        elif len(df) == 1:
            prev_close = float(df.iloc[-1]["Open"])
            cur_close = float(df.iloc[-1]["Close"])
        else:
            prev_close = current_price
            cur_close = current_price

        # 2. ARA/ARB
        health["ara_arb"] = compute_ara_arb(cur_close, prev_close)
    except Exception as e:
        health["yfinance"] = {"error": str(e)}
        health["ara_arb"] = {"error": str(e)}

    # 3. DB freshness
    if db_session:
        try:
            from sqlalchemy import text
            r = await db_session.execute(
                text("SELECT max(date), count(*) FROM ohlcv_daily WHERE ticker=:t"),
                {"t": ticker if ticker.endswith(".JK") else f"{ticker}.JK"}
            )
            row = r.fetchone()
            if row and row[0]:
                last_date = row[0]
                days_old = (date.today() - last_date).days if last_date else 999
                health["db_freshness"] = {
                    "last_date": str(last_date),
                    "row_count": row[1],
                    "days_old": days_old,
                    "status": "ok" if days_old <= DATA_STALE_DAYS else "stale"
                }
            else:
                health["db_freshness"] = {"status": "no_data", "row_count": 0}
        except Exception as e:
            health["db_freshness"] = {"error": str(e)}

    return health
