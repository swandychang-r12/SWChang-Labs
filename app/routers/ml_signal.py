"""
ml_signal.py — /api/v2/signal/{ticker}
XGBoost v6 walk-forward model, AUC_WF=0.5975
Phase 3: C-09 | 2026-06-30
"""
import asyncio
import logging
import sys
from datetime import date
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException

from app.utils import api_response

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v2", tags=["ml-signal"])

MODEL_FILE = "xgb_v6_20260630-1209_7744.joblib"
AUC_WF = 0.5975

# Auto-detect path: inside Docker = /trading-scanner, outside = /home/aiops/trading-scanner
if Path("/trading-scanner/models").exists():
    SCANNER_PATH = Path("/trading-scanner")
else:
    SCANNER_PATH = Path("/home/aiops/trading-scanner")


@lru_cache(maxsize=1)
def _load_v6():
    """Load xgb_v6 bundle once at startup, cache forever."""
    spath = str(SCANNER_PATH)
    if spath not in sys.path:
        sys.path.insert(0, spath)
    try:
        import joblib
        model_path = SCANNER_PATH / "models" / MODEL_FILE
        if not model_path.exists():
            logger.warning(f"xgb_v6 not found: {model_path}")
            return None
        bundle = joblib.load(model_path)
        feat_count = len(bundle.get("features", [])) if isinstance(bundle, dict) else "N/A"
        logger.info(f"[ml_signal] xgb_v6 loaded. Features: {feat_count}")
        return bundle
    except Exception as e:
        logger.error(f"[ml_signal] xgb_v6 load failed: {e}")
        return None


def _compute_signal(ticker: str) -> dict:
    bundle = _load_v6()
    if bundle is None:
        raise HTTPException(status_code=503, detail="xgb_v6 model unavailable — check model path")

    # Bundle is dict with keys: model, features, params
    clf = bundle["model"]
    feature_names = bundle.get("features", [])

    features_path = SCANNER_PATH / "data" / "features.parquet"
    if not features_path.exists():
        raise HTTPException(status_code=503, detail=f"features.parquet not found at {features_path}")

    df = pd.read_parquet(features_path)

    # Normalize ticker — try both with and without .JK suffix
    ticker_plain = ticker.upper().replace(".JK", "")
    ticker_jk = ticker_plain + ".JK"

    if "ticker" in df.columns:
        mask = df["ticker"].isin([ticker_plain, ticker_jk])
    else:
        mask = df.index.isin([ticker_plain, ticker_jk])

    ticker_df = df[mask].tail(1)
    if ticker_df.empty:
        raise HTTPException(status_code=404, detail=f"Ticker {ticker_plain} not found in features.parquet")

    # Determine as_of date from data
    as_of = date.today().isoformat()
    if "date" in ticker_df.columns:
        try:
            as_of = str(ticker_df["date"].iloc[0])[:10]
        except Exception:
            pass

    # Build feature matrix using exact feature list from bundle
    available = [f for f in feature_names if f in ticker_df.columns]
    if not available:
        # Fallback: drop all non-numeric/meta columns
        meta_cols = [c for c in ["ticker", "date", "label_win", "target", "label"] if c in ticker_df.columns]
        X = ticker_df.drop(columns=meta_cols, errors="ignore").select_dtypes(include=["number"])
    else:
        X = ticker_df[available]

    prob = float(clf.predict_proba(X)[0][1])
    signal = 1 if prob >= 0.5 else 0

    return {
        "ticker": ticker_plain,
        "signal": signal,
        "probability": round(prob, 4),
        "model": "xgb_v6",
        "auc_wf": AUC_WF,
        "as_of": as_of,
    }


@router.get("/signal/{ticker}", summary="ML signal for a ticker (xgb_v6)")
async def get_signal(ticker: str):
    """
    Returns XGBoost v6 walk-forward signal for the given ticker.

    - **signal**: 1 = buy signal, 0 = no signal
    - **probability**: model confidence (0.0–1.0)
    - **auc_wf**: walk-forward AUC of this model version
    - **as_of**: date of the latest feature row used
    """
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, lambda: _compute_signal(ticker))
        return api_response(True, data=result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[ml_signal] error for {ticker}: {e}", exc_info=True)
        return api_response(False, error=str(e))