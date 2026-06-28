from fastapi import APIRouter, Depends, Body
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from typing import List, Dict, Any
from datetime import datetime
from app.database import get_db
from app.models.trade import BacktestRun
from app.services.backtest_svc import run_backtest
from app.utils import api_response

router = APIRouter(prefix="/api/backtest", tags=["backtest"])

class BacktestRequest(BaseModel):
    strategy: str = "ml_xgboost_v5"
    tickers: List[str]
    start_date: str = "2023-01-01"
    end_date: str = "auto"
    capital: float = 100000000
    params: Dict[str, Any] = {}

@router.post("")
async def execute_backtest(
    req: BacktestRequest = Body(...),
    db: AsyncSession = Depends(get_db),
):
    try:
        end = req.end_date if req.end_date != "auto" else datetime.now().strftime("%Y-%m-%d")
        results = await run_backtest(
            req.strategy, req.tickers, req.start_date, end, req.capital, req.params
        )
        if results.get("error"):
            return api_response(False, error=results["error"])
        # Persist
        record = BacktestRun(
            strategy=req.strategy,
            tickers=req.tickers,
            start_date=datetime.fromisoformat(req.start_date).date(),
            end_date=datetime.fromisoformat(end).date(),
            initial_capital=req.capital,
            final_capital=results.get("final_capital"),
            total_return_pct=results.get("total_return_pct"),
            max_drawdown_pct=results.get("max_drawdown_pct"),
            sharpe_ratio=results.get("sharpe_ratio"),
            win_rate=results.get("win_rate"),
            total_trades=results.get("total_trades"),
            results=results,
        )
        db.add(record)
        await db.commit()
        await db.refresh(record)
        return api_response(True, data={**results, "id": record.id})
    except Exception as e:
        return api_response(False, error=str(e))
