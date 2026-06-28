from datetime import datetime, date
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db, JournalTrade, PostTradeReview
from app.agents.orchestrator import OrchestratorAgent

router = APIRouter(prefix="/api/journal", tags=["journal"])

@router.post("/entry")
async def create_journal_entry(
    ticker: str,
    side: str,
    entry_price: float,
    stop_loss: Optional[float] = None,
    target_price: Optional[float] = None,
    lot_size: int = 1,
    notes: Optional[str] = None,
    strategy_used: Optional[str] = None,
    signal_date: Optional[date] = None,
    db: AsyncSession = Depends(get_db)
):
    """Create a new journal entry"""
    if side not in ["BUY", "SELL"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Side must be either BUY or SELL"
        )
    if signal_date is None:
        signal_date = datetime.now().date()
    new_entry = JournalTrade(
        ticker=ticker.upper(),
        side=side,
        entry_price=entry_price,
        stop_loss=stop_loss,
        target_price=target_price,
        lot_size=lot_size,
        notes=notes,
        strategy_used=strategy_used,
        signal_date=signal_date
    )
    db.add(new_entry)
    await db.commit()
    await db.refresh(new_entry)
    return {"success": True, "message": "Journal entry created", "data": new_entry}

@router.get("/")
async def get_journal_entries(
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db)
):
    """Get journal entries with pagination"""
    stmt = select(JournalTrade).order_by(desc(JournalTrade.signal_date)).limit(limit).offset(offset)
    result = await db.execute(stmt)
    entries = result.scalars().all()
    return {"success": True, "data": entries, "limit": limit, "offset": offset}

@router.put("/{id}")
async def update_journal_entry(
    id: int,
    exit_price: Optional[float] = None,
    exit_date: Optional[date] = None,
    notes: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """Update a journal entry (exit details)"""
    stmt = select(JournalTrade).where(JournalTrade.id == id)
    result = await db.execute(stmt)
    entry = result.scalars().first()
    if not entry:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Journal entry not found")
    if exit_price is not None:
        entry.exit_price = exit_price
    if exit_date is not None:
        entry.exit_date = exit_date
    if notes is not None:
        entry.notes = notes
    if exit_price is not None and entry.entry_price is not None:
        if entry.side == "BUY":
            entry.pnl_pct = (exit_price - entry.entry_price) / entry.entry_price * 100
        else:
            entry.pnl_pct = (entry.entry_price - exit_price) / entry.entry_price * 100
        entry.pnl_idr = entry.pnl_pct * entry.entry_price * entry.lot_size / 100
    entry.updated_at = datetime.now()
    await db.commit()
    await db.refresh(entry)
    return {"success": True, "message": "Journal entry updated", "data": entry}

@router.get("/stats")
async def get_journal_stats(db: AsyncSession = Depends(get_db)):
    """Get journal statistics"""
    total_trades = await db.scalar(select(func.count(JournalTrade.id)))
    win_trades = await db.scalar(select(func.count(JournalTrade.id)).where(JournalTrade.pnl_pct > 0))
    win_rate = round(win_trades / total_trades * 100, 2) if total_trades > 0 else 0
    avg_rr = await db.scalar(
        select(func.avg(JournalTrade.pnl_pct / (func.abs(JournalTrade.stop_loss - JournalTrade.entry_price) / JournalTrade.entry_price * 100 if JournalTrade.stop_loss else 1)))
        .where(JournalTrade.pnl_pct > 0, JournalTrade.stop_loss.isnot(None))
    )
    avg_rr = round(avg_rr, 2) if avg_rr is not None else 0
    best_trade = await db.scalar(select(JournalTrade).order_by(desc(JournalTrade.pnl_pct)).limit(1))
    worst_trade = await db.scalar(select(JournalTrade).order_by(JournalTrade.pnl_pct).limit(1))
    total_pnl_idr = await db.scalar(select(func.sum(JournalTrade.pnl_idr)).where(JournalTrade.pnl_idr.isnot(None)))
    total_pnl_idr = round(total_pnl_idr, 2) if total_pnl_idr is not None else 0
    monthly_breakdown = await db.execute(
        select(
            func.to_char(JournalTrade.signal_date, 'YYYY-MM').label('month'),
            func.sum(JournalTrade.pnl_idr).label('monthly_pnl'),
            func.count(JournalTrade.id).label('trade_count')
        ).group_by('month').order_by('month')
    )
    monthly_breakdown = [
        {"month": row.month, "pnl_idr": round(row.monthly_pnl, 2), "trade_count": row.trade_count}
        for row in monthly_breakdown
    ]
    return {"success": True, "data": {"total_trades": total_trades, "win_rate": win_rate, "avg_rr": avg_rr, "best_trade": best_trade, "worst_trade": worst_trade, "total_pnl_idr": total_pnl_idr, "monthly_breakdown": monthly_breakdown}}

@router.post("/{id}/post-trade-review")
async def create_post_trade_review(
    id: int,
    what_happened: str,
    what_worked: Optional[str] = None,
    what_to_change: Optional[str] = None,
    emotion_score: Optional[int] = None,
    db: AsyncSession = Depends(get_db)
):
    """Create post-trade review and trigger AI summary"""
    stmt = select(JournalTrade).where(JournalTrade.id == id)
    result = await db.execute(stmt)
    trade = result.scalars().first()
    if not trade:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Trade not found")
    review = PostTradeReview(
        trade_id=id,
        what_happened=what_happened,
        what_worked=what_worked,
        what_to_change=what_to_change,
        emotion_score=emotion_score
    )
    db.add(review)
    await db.commit()
    await db.refresh(review)
    orchestrator = OrchestratorAgent(
        name="orchestrator",
        role="Chief Investment Officer",
        model="qwen2.5:7b",
        temperature=0.7,
        max_tokens=1024,
        gateway_url="http://localhost:11434/v1",
        api_key="ollama"
    )
    context = {
        "trade": {
            "ticker": trade.ticker,
            "side": trade.side,
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "pnl_pct": trade.pnl_pct,
            "strategy_used": trade.strategy_used,
            "signal_date": trade.signal_date.isoformat() if trade.signal_date else None,
            "exit_date": trade.exit_date.isoformat() if trade.exit_date else None
        },
        "review": {
            "what_happened": what_happened,
            "what_worked": what_worked,
            "what_to_change": what_to_change,
            "emotion_score": emotion_score
        }
    }
    summary = await orchestrator.analyze(
        ticker=trade.ticker,
        market_data={"context": context, "task": "Generate concise post-trade review summary"}
    )
    review.ai_summary = summary.get("reasoning", "No summary generated")
    await db.commit()
    await db.refresh(review)
    return {"success": True, "message": "Post-trade review created", "data": review}
