from datetime import datetime, timezone
from typing import Any, Optional, Dict

def api_response(success: bool, data: Any = None, error: Optional[str] = None) -> Dict:
    resp: Dict[str, Any] = {
        "success": success,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    if success:
        resp["data"] = data
    else:
        resp["error"] = error
    return resp
