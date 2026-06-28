from fastapi import APIRouter
from app.services.broker_flow_svc import get_paper_state
from app.utils import api_response

router = APIRouter(prefix="/api/paper-trade", tags=["paper-trade"])

@router.get("")
async def get_paper_trading():
    try:
        state = await get_paper_state()
        return api_response(True, data=state)
    except Exception as e:
        return api_response(False, error=str(e))
