from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.agents.debate_engine import DebateEngine
from app.services.data_fetcher import fetch_market_data

router = APIRouter(prefix="/api/debate", tags=["debate"])

@router.post("/{ticker}")
async def run_debate_analysis(
    ticker: str,
    force_refresh: bool = False,
    db: AsyncSession = Depends(get_db)
):
    """
    Run multi-agent debate analysis for a given ticker.
    
    Args:
        ticker: Stock ticker (e.g., BBCA)
        force_refresh: If True, bypass cache and run fresh analysis
        db: Database session
    
    Returns:
        Complete debate analysis result
    """
    try:
        # Fetch market data first
        market_data = await fetch_market_data(ticker)
        if not market_data:
            raise HTTPException(status_code=404, detail=f"Market data not found for {ticker}")

        # Run the debate engine
        result = await DebateEngine.run_debate(
            ticker=ticker,
            market_data=market_data,
            db_session=db,
            force_refresh=force_refresh
        )

        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "ticker": ticker,
            "result": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Debate analysis failed: {str(e)}")

@router.get("/{ticker}/history")
async def get_debate_history(
    ticker: str,
    limit: int = 10,
    db: AsyncSession = Depends(get_db)
):
    """
    Get debate analysis history for a ticker.
    
    Args:
        ticker: Stock ticker
        limit: Maximum number of historical analyses to return
    
    Returns:
        List of historical analyses
    """
    from sqlalchemy import select
    
    stmt = select(AIAnalysis).where(
        AIAnalysis.ticker == ticker
    ).order_by(AIAnalysis.created_at.desc()).limit(limit)
    
    result = await db.execute(stmt)
    analyses = result.scalars().all()
    
    return {
        "success": True,
        "ticker": ticker,
        "history": [
            {
                "id": analysis.id,
                "analysis_date": analysis.analysis_date.isoformat(),
                "action": analysis.action,
                "confidence": float(analysis.confidence),
                "created_at": analysis.created_at.isoformat()
            }
            for analysis in analyses
        ]
    }
