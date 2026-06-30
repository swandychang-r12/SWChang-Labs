"""
ml_paper_service.py -- ML Paper Trading engine
Phase 4: C-10 | 2026-06-30
Model: xgb_v6 | AUC_WF: 0.5975
"""
import asyncio
import logging
import statistics
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from sqlalchemy import (
    Column, Integer, String, Date, Float, TIMESTAMP, Index,
    select, and_, func as sqlfunc
)
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import Base, AsyncSessionLocal

logger = logging.getLogger(__name__)

SCANNER_PATH = Path("/trading-scanner") if Path("/trading-scanner").exists() else Path("/home/aiops/trading-scanner")
MAX_OPEN_TRADES = 5
MODEL_NAME = "xgb_v6"
AUC_WF = 0.5975


# ORM Model
class MLPaperTrade(Base):
    __tablename__ = "ml_paper_trades"
    __table_args__ = (Index("ix_ml_paper_ticker_entry", "ticker", "entry_date"),)

    id          = Column(Integer, primary_key=True, autoincrement=True)
    ticker      = Column(String(12), nullable=False, index=True)
    entry_date  = Column(Date, nullable=False)
    entry_price = Column(Float, nullable=False)
    sl_price    = Column(Float, nullable=False)      # entry - 2*ATR14
    signal_prob = Column(Float, nullable=False)
    exit_date   = Column(Date, nullable=True)
    exit_price  = Column(Float, nullable=True)
    return_pct  = Column(Float, nullable=True)
    exit_reason = Column(String(10), nullable=True)  # T3 | SL
    status      = Column(String(8), nullable=False, default="open")  # open|closed|stopped
    created_at  = Column(TIMESTAMP(timezone=True), server_default=sqlfunc.now())


def _t3_exit_date(entry: date) -> date:
    """Return T+3 trading days (skip Sat/Sun)."""
    d, count = entry, 0
    while count < 3:
        d += timedelta(days=1)
        if d.weekday() < 5:
            count += 1
    return d


def _bulk_scan_sync() -> list:
    """
    Vectorised scan: latest row per ticker -> model -> sorted results.
    Returns list of dicts with keys: ticker, signal, probability, as_of, close, atr14.
    CPU-bound -- call via run_in_executor.
    """
    from app.routers.ml_signal import _load_v6
    bundle = _load_v6()
    if bundle is None:
        logger.warning("[scan] xgb_v6 not loaded")
        return []

    clf = bundle["model"]
    feature_names = bundle.get("features", [])

    features_path = SCANNER_PATH / "data" / "features.parquet"
    if not features_path.exists():
        logger.warning(f"[scan] features.parquet not found at {features_path}")
        return []

    df = pd.read_parquet(features_path)

    # One row per ticker (latest date)
    latest = (
        df.sort_values("date")
        .groupby("ticker", as_index=False)
        .last()
        .reset_index(drop=True)
    )

    available = [f for f in feature_names if f in latest.columns]
    X = latest[available].fillna(0)
    probs = clf.predict_proba(X)[:, 1]

    results = []
    for idx, row in latest.iterrows():
        ticker_plain = str(row.get("ticker", "")).replace(".JK", "").upper()
        prob = float(probs[idx])
        results.append({
            "ticker":      ticker_plain,
            "signal":      1 if prob >= 0.5 else 0,
            "probability": round(prob, 4),
            "as_of":       str(row.get("date", ""))[:10],
            "close":       float(row.get("close", 0) or 0),
            "atr14":       float(row.get("atr14", 0) or 0),
        })

    return sorted(results, key=lambda x: x["probability"], reverse=True)


async def scan_all() -> dict:
    """Async wrapper for bulk scan. Returns API-ready dict."""
    loop = asyncio.get_event_loop()
    all_results = await loop.run_in_executor(None, _bulk_scan_sync)
    buy_signals = [r for r in all_results if r["signal"] == 1]
    # Strip internal-only fields from API response
    clean = [{k: v for k, v in r.items() if k not in ("close", "atr14")} for r in buy_signals]
    return {
        "signals":       clean,
        "scan_date":     str(date.today()),
        "total_buy":     len(buy_signals),
        "total_scanned": len(all_results),
    }


async def get_open_count(db: AsyncSession) -> int:
    r = await db.execute(
        select(sqlfunc.count()).select_from(MLPaperTrade).where(MLPaperTrade.status == "open")
    )
    return r.scalar() or 0


async def create_paper_trades(scan_results: list, db: AsyncSession) -> list:
    """
    Create paper trade entries for signal=1 tickers.
    Skips: existing open trade for ticker | already entered today | max open reached.
    """
    today = date.today()
    open_count = await get_open_count(db)
    created = []

    for r in scan_results:
        if r["signal"] != 1:
            continue
        if open_count >= MAX_OPEN_TRADES:
            logger.info(f"[paper] max open={MAX_OPEN_TRADES} reached")
            break

        # Already open?
        ex = await db.execute(
            select(MLPaperTrade).where(
                and_(MLPaperTrade.ticker == r["ticker"], MLPaperTrade.status == "open")
            )
        )
        if ex.scalars().first():
            continue

        # Already entered today?
        td = await db.execute(
            select(MLPaperTrade).where(
                and_(MLPaperTrade.ticker == r["ticker"], MLPaperTrade.entry_date == today)
            )
        )
        if td.scalars().first():
            continue

        entry_price = r.get("close", 0)
        atr14 = r.get("atr14", 0)
        if entry_price <= 0:
            logger.warning(f"[paper] no valid entry price for {r['ticker']}, skipping")
            continue

        sl = entry_price - 2.0 * atr14 if atr14 > 0 else entry_price * 0.95
        trade = MLPaperTrade(
            ticker=r["ticker"],
            entry_date=today,
            entry_price=round(entry_price, 2),
            sl_price=round(sl, 2),
            signal_prob=r["probability"],
            status="open",
        )
        db.add(trade)
        open_count += 1
        created.append(r["ticker"])

    await db.commit()
    return created


async def fill_ml_outcomes(db: AsyncSession = None) -> int:
    """
    Fill exit for open trades that hit T+3 or SL.
    Standalone (db=None) or within existing session.
    """
    own_session = db is None
    if own_session:
        db = AsyncSessionLocal()

    try:
        today = date.today()
        result = await db.execute(
            select(MLPaperTrade).where(MLPaperTrade.status == "open")
        )
        open_trades = result.scalars().all()
        if not open_trades:
            return 0

        # Batch price fetch via yfinance
        prices: dict = {}
        try:
            import yfinance as yf
            tickers_jk = list({f"{t.ticker}.JK" for t in open_trades})
            data = yf.download(tickers_jk, period="5d", interval="1d",
                               progress=False, auto_adjust=True)
            close_col = data.get("Close", data) if hasattr(data, "get") else data
            if len(tickers_jk) == 1:
                s = close_col.dropna()
                if len(s):
                    prices[tickers_jk[0]] = float(s.iloc[-1])
            else:
                for tkr in tickers_jk:
                    try:
                        s = close_col[tkr].dropna()
                        if len(s):
                            prices[tkr] = float(s.iloc[-1])
                    except Exception:
                        pass
        except Exception as e:
            logger.error(f"[outcome] yfinance error: {e}")

        closed_n = 0
        for trade in open_trades:
            t3 = _t3_exit_date(trade.entry_date)
            cur = prices.get(f"{trade.ticker}.JK")
            hit_sl = bool(cur and cur <= trade.sl_price)
            due_t3 = today >= t3

            if not (hit_sl or due_t3):
                continue
            if cur is None:
                continue

            trade.exit_price  = round(cur, 2)
            trade.exit_date   = today
            trade.return_pct  = round((cur / trade.entry_price - 1) * 100, 4)
            trade.exit_reason = "SL" if hit_sl and not due_t3 else "T3"
            trade.status      = "stopped" if trade.exit_reason == "SL" else "closed"
            closed_n += 1

        await db.commit()
        logger.info(f"[outcome] ml_paper closed {closed_n} trades")
        return closed_n

    finally:
        if own_session:
            await db.close()


async def get_stats(db: AsyncSession) -> dict:
    """Aggregate stats from all closed/stopped trades."""
    r = await db.execute(
        select(MLPaperTrade).where(MLPaperTrade.status.in_(["closed", "stopped"]))
    )
    closed = r.scalars().all()
    open_n = await get_open_count(db)

    if not closed:
        return {
            "total_trades": 0, "open_trades": open_n,
            "win_rate": None, "avg_return_pct": None, "sharpe": None,
            "model": MODEL_NAME, "auc_wf": AUC_WF,
            "note": "No closed trades yet -- run /api/v2/paper/run-scan first",
        }

    returns = [t.return_pct for t in closed if t.return_pct is not None]
    wins = [r for r in returns if r > 0]
    avg_r = round(sum(returns) / len(returns), 4) if returns else None
    wr = round(len(wins) / len(returns) * 100, 2) if returns else None

    sharpe = None
    if len(returns) >= 3:
        std = statistics.stdev(returns)
        if std > 0:
            sharpe = round((sum(returns) / len(returns)) / std, 3)

    return {
        "total_trades":   len(closed),
        "open_trades":    open_n,
        "win_rate":       wr,
        "avg_return_pct": avg_r,
        "sharpe":         sharpe,
        "sl_trades":      sum(1 for t in closed if t.exit_reason == "SL"),
        "t3_trades":      sum(1 for t in closed if t.exit_reason == "T3"),
        "model":          MODEL_NAME,
        "auc_wf":         AUC_WF,
    }