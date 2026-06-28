from fastapi import APIRouter, Path, Query
from app.services.broker_flow_svc import get_broker_flow
from app.utils import api_response

router = APIRouter(prefix="/api/broker-flow", tags=["broker-flow"])

@router.get("/{ticker}")
async def broker_flow(
    ticker: str = Path(...),
    days: int = Query(10, ge=1, le=90),
):
    try:
        data = await get_broker_flow(ticker.upper(), days)
        return api_response(True, data=data)
    except Exception as e:
        return api_response(False, error=str(e))
