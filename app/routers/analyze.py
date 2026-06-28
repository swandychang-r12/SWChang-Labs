from fastapi import APIRouter, Depends, Path, Body
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
from app.database import get_db
from app.services.data_fetcher import fetch_ohlcv
from app.utils import api_response

router = APIRouter(prefix="/api/analyze", tags=["analysis"])

@router.post("/{ticker}")
async def analyze_stock(
    ticker: str = Path(..., description="IDX ticker e.g. BBCA"),
    force_refresh: bool = Body(False, embed=True),
    db: AsyncSession = Depends(get_db),
):
    try:
        ohlcv = await fetch_ohlcv(ticker.upper(), period="60d")
        if not ohlcv or "error" in ohlcv:
            return api_response(False, error=f"No data for {ticker}: {ohlcv}")
        return api_response(True, data={
            "ticker": ticker.upper(),
            "ohlcv": ohlcv,
            "note": "Phase 2 — AI agent debate will be added here",
        })
    except Exception as e:
        return api_response(False, error=str(e))

@router.get("/{ticker}/ohlcv")
async def get_ohlcv(
    ticker: str = Path(...),
    period: str = "60d",
    interval: str = "1d",
    db: AsyncSession = Depends(get_db),
):
    try:
        data = await fetch_ohlcv(ticker.upper(), period=period, interval=interval)
        if not data:
            return api_response(False, error=f"No data for {ticker}")
        return api_response(True, data=data)
    except Exception as e:
        return api_response(False, error=str(e))
