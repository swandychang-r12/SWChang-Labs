from pydantic_settings import BaseSettings
from pydantic import Field
import yaml
from typing import Dict, Any

class Settings(BaseSettings):
    database_url: str = Field(
        default="postgresql+asyncpg://r12user:a7fe679a67d8fa6b5953d5471be37395870db5bf@r12-postgres:5432/swandy_fund"
    )
    trading_scanner_path: str = Field(default="/trading-scanner")
    config_yaml_path: str = Field(default="/app/config.yaml")
    ohlcv_cache_ttl_seconds: int = 300
    api_host: str = "0.0.0.0"
    api_port: int = 8089

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()

def load_config_yaml() -> Dict[str, Any]:
    try:
        with open(settings.config_yaml_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}
