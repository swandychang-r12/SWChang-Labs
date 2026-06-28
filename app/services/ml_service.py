"""
ml_service.py — Bridge ke XGBoost v5 di trading-scanner
TIDAK memodifikasi trading-scanner, hanya import dan wrap.
"""
import sys
import asyncio
import logging
from pathlib import Path
from functools import lru_cache
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)

# Auto-detect path: inside Docker = /trading-scanner, outside = /home/aiops/trading-scanner
if Path("/trading-scanner/models").exists():
    SCANNER_PATH = Path("/trading-scanner")
else:
    SCANNER_PATH = Path("/home/aiops/trading-scanner")


def _ensure_scanner_path():
    str_path = str(SCANNER_PATH)
    if str_path not in sys.path:
        sys.path.insert(0, str_path)


@lru_cache(maxsize=1)
def _load_model():
    """Load XGBoost model bundle sekali, cache selamanya.
    Returns dict: {'model': XGBClassifier, 'features': [...], ...} or None.
    """
    _ensure_scanner_path()
    try:
        import joblib
        model_path = SCANNER_PATH / "models" / "xgb_v5_cls.joblib"
        if not model_path.exists():
            logger.warning(f"XGBoost model not found at {model_path}")
            return None
        bundle = joblib.load(model_path)
        logger.info(f"XGBoost v5 loaded. Features: {len(bundle.get('features', []))}")
        return bundle
    except Exception as e:
        logger.error(f"Failed to load XGBoost: {e}")
        return None


async def get_ml_probability(ticker: str) -> dict:
    """
    Ambil XGBoost v5 probability untuk ticker.
    Return: {
        "probability": float,  # 0.0-1.0, terkalibrasi dari backtest
        "signal": "STRONG_BUY|BUY|NEUTRAL|SELL|STRONG_SELL",
        "available": bool,
        "source": "xgb_v5|unavailable|error"
    }
    """
    def _compute():
        bundle = _load_model()
        if bundle is None:
            return {"probability": 0.5, "signal": "NEUTRAL", "available": False, "source": "unavailable"}

        try:
            clf = bundle["model"]
            feature_names = bundle.get("features", [])

            features_path = SCANNER_PATH / "data" / "features.parquet"
            if not features_path.exists():
                return {"probability": 0.5, "signal": "NEUTRAL", "available": False, "source": "unavailable"}

            df = pd.read_parquet(features_path)
            ticker_jk = ticker if ticker.endswith(".JK") else f"{ticker}.JK"

            # Find ticker row
            if "ticker" in df.columns:
                mask = df["ticker"].isin([ticker, ticker_jk])
            elif df.index.name == "ticker" or (hasattr(df.index, "name") and "ticker" in str(df.index.name)):
                mask = df.index.isin([ticker, ticker_jk])
            else:
                mask = df.index.isin([ticker, ticker_jk])

            ticker_row = df[mask].tail(1)
            if ticker_row.empty:
                return {"probability": 0.5, "signal": "NEUTRAL", "available": False, "source": "unavailable"}

            # Select features in the exact order the model expects
            if feature_names:
                available = [f for f in feature_names if f in ticker_row.columns]
                X = ticker_row[available]
            else:
                # Fallback: drop non-feature columns
                drop_cols = [c for c in ["ticker", "date", "target", "label"] if c in ticker_row.columns]
                X = ticker_row.drop(columns=drop_cols).select_dtypes(include=["number"])

            prob = float(clf.predict_proba(X)[0][1])

            if prob >= 0.70:
                signal = "STRONG_BUY"
            elif prob >= 0.58:
                signal = "BUY"
            elif prob >= 0.45:
                signal = "NEUTRAL"
            elif prob >= 0.35:
                signal = "SELL"
            else:
                signal = "STRONG_SELL"

            return {
                "probability": round(prob, 4),
                "signal": signal,
                "available": True,
                "source": "xgb_v5"
            }
        except Exception as e:
            logger.warning(f"XGBoost inference failed for {ticker}: {e}")
            return {"probability": 0.5, "signal": "NEUTRAL", "available": False, "source": "error"}

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _compute)
