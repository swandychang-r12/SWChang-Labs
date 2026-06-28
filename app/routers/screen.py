from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
from app.database import get_db
from app.services.scanner import run_scanner
from app.config import load_config_yaml
from app.utils import api_response

router = APIRouter(prefix="/api/screen", tags=["screener"])

@router.get("")
async def screen_stocks(
    universe: Optional[str] = Query(None, description="lq45 | all | comma-separated tickers"),
    min_score: float = Query(0.52, ge=0, le=1),
    min_volume_ratio: float = Query(1.5, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    try:
        cfg = load_config_yaml()
        default_universe = cfg.get("trading", {}).get("universe", [])
        if universe and universe not in ("all", "lq45"):
            tickers = [t.strip().upper() for t in universe.split(",")]
        else:
            tickers = default_universe
        results = await run_scanner(tickers, min_score, min_volume_ratio, limit)
        return api_response(True, data={"count": len(results), "results": results})
    except Exception as e:
        return api_response(False, error=str(e))
