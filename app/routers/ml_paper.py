"""
ml_paper.py -- /api/v2/paper/* endpoints
Phase 4: C-10 | 2026-06-30
"""
import asyncio
import logging
from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database import get_db
from app.utils import api_response
from app.services.ml_paper_service import (
    MLPaperTrade, scan_all, create_paper_trades,
    fill_ml_outcomes, get_stats, _bulk_scan_sync,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v2/paper", tags=["ml-paper"])


@router.get("/trades", summary="List ML paper trades")
async def list_trades(
    status: Optional[str] = Query("all", description="open | closed | stopped | all"),
    limit: int = Query(100, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
):
    try:
        q = select(MLPaperTrade).order_by(MLPaperTrade.entry_date.desc()).limit(limit)
        if status and status != "all":
            q = q.where(MLPaperTrade.status == status)
        result = await db.execute(q)
        trades = result.scalars().all()
        data = [
            {
                "id":          t.id,
                "ticker":      t.ticker,
                "entry_date":  str(t.entry_date),
                "entry_price": t.entry_price,
                "sl_price":    t.sl_price,
                "signal_prob": t.signal_prob,
                "exit_date":   str(t.exit_date) if t.exit_date else None,
                "exit_price":  t.exit_price,
                "return_pct":  t.return_pct,
                "exit_reason": t.exit_reason,
                "status":      t.status,
            }
            for t in trades
        ]
        return api_response(True, data={"trades": data, "count": len(data), "filter": status})
    except Exception as e:
        logger.error(f"[paper/trades] {e}", exc_info=True)
        return api_response(False, error=str(e))


@router.get("/stats", summary="ML paper trading performance stats")
async def paper_stats(db: AsyncSession = Depends(get_db)):
    try:
        stats = await get_stats(db)
        return api_response(True, data=stats)
    except Exception as e:
        logger.error(f"[paper/stats] {e}", exc_info=True)
        return api_response(False, error=str(e))


@router.post("/run-scan", summary="Trigger ML scan + create paper trades for today")
async def run_scan(db: AsyncSession = Depends(get_db)):
    """
    1. Scan all 80 tickers via xgb_v6 (vectorised)
    2. Create paper trades for signal=1 tickers (max 5 open)
    3. Fill any open trades that hit T+3 or SL
    """
    try:
        loop = asyncio.get_event_loop()
        all_results = await loop.run_in_executor(None, _bulk_scan_sync)
        buy_signals = [r for r in all_results if r["signal"] == 1]
        created  = await create_paper_trades(all_results, db)
        closed_n = await fill_ml_outcomes(db)
        return api_response(True, data={
            "scan_date":     str(date.today()),
            "total_scanned": len(all_results),
            "total_buy":     len(buy_signals),
            "trades_created": created,
            "trades_closed":  closed_n,
            "top_signals": [
                {"ticker": r["ticker"], "probability": r["probability"]}
                for r in buy_signals[:10]
            ],
        })
    except Exception as e:
        logger.error(f"[paper/run-scan] {e}", exc_info=True)
        return api_response(False, error=str(e))