from fastapi import APIRouter, Body
from typing import Any, Dict
import yaml
from app.config import settings
from app.utils import api_response

router = APIRouter(prefix="/api/config", tags=["config"])

@router.get("")
async def get_config():
    try:
        with open(settings.config_yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return api_response(True, data=cfg)
    except Exception as e:
        return api_response(False, error=str(e))

@router.put("")
async def update_config(updates: Dict[str, Any] = Body(...)):
    try:
        with open(settings.config_yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        _deep_merge(cfg, updates)
        with open(settings.config_yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)
        return api_response(True, data={"message": "Config updated"})
    except Exception as e:
        return api_response(False, error=str(e))

def _deep_merge(base: dict, updates: dict):
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
