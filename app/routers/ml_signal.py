"""
ml_signal.py - /api/v2/signal/{ticker} + /api/v2/model/info
Auto-retrain aware: reads model from current_model.json pointer.
Phase 3 updated | 2026-06-30
"""
import asyncio
import json
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException

from app.utils import api_response

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v2", tags=["ml-signal"])

# Auto-detect scanner path
if Path("/trading-scanner/models").exists():
    SCANNER_PATH = Path("/trading-scanner")
else:
    SCANNER_PATH = Path("/home/aiops/trading-scanner")

CURRENT_PTR = SCANNER_PATH / "models" / "current_model.json"

# Module-level cache: (model_file_name, bundle_dict)
_model_cache: tuple = (None, None)


def _read_ptr() -> dict:
    if CURRENT_PTR.exists():
        with open(CURRENT_PTR) as f:
            return json.load(f)
    return {
        "model_file": "xgb_v6_20260630-1209_7744.joblib",
        "auc_wf": 0.5975,
    }


def _load_v6():
    """Load model from current_model.json pointer. Re-loads if pointer changed."""
    global _model_cache
    ptr = _read_ptr()
    model_file = ptr.get("model_file")

    cached_file, cached_bundle = _model_cache
    if cached_file == model_file and cached_bundle is not None:
        return cached_bundle, ptr

    spath = str(SCANNER_PATH)
    if spath not in sys.path:
        sys.path.insert(0, spath)
    try:
        import joblib
        model_path = SCANNER_PATH / "models" / model_file
        if not model_path.exists():
            logger.warning(f"Model not found: {model_path}")
            return None, ptr
        bundle = joblib.load(model_path)
        _model_cache = (model_file, bundle)
        feat_count = len(bundle.get("features", [])) if isinstance(bundle, dict) else "N/A"
        logger.info(f"[ml_signal] Loaded {model_file} | features={feat_count} | auc_wf={ptr.get('auc_wf')}")
        return bundle, ptr
    except Exception as e:
        logger.error(f"[ml_signal] Load failed: {e}")
        return None, ptr


def _preload():
    """Call at startup (from main.py lifespan)."""
    _load_v6()


def _compute_signal(ticker: str) -> dict:
    bundle, ptr = _load_v6()
    if bundle is None:
        raise HTTPException(status_code=503, detail="ML model unavailable")

    clf = bundle["model"]
    feature_names = bundle.get("features", [])

    features_path = SCANNER_PATH / "data" / "features.parquet"
    if not features_path.exists():
        raise HTTPException(status_code=503, detail="features.parquet not found")

    df = pd.read_parquet(features_path)

    ticker_plain = ticker.upper().replace(".JK", "")
    ticker_jk    = ticker_plain + ".JK"

    if "ticker" in df.columns:
        mask = df["ticker"].isin([ticker_plain, ticker_jk])
    else:
        mask = df.index.isin([ticker_plain, ticker_jk])

    ticker_df = df[mask].tail(1)
    if ticker_df.empty:
        raise HTTPException(status_code=404, detail=f"Ticker {ticker_plain} not found")

    as_of = date.today().isoformat()
    if "date" in ticker_df.columns:
        try:
            as_of = str(ticker_df["date"].iloc[0])[:10]
        except Exception:
            pass

    available = [f for f in feature_names if f in ticker_df.columns]
    if not available:
        meta_cols = [c for c in ["ticker", "date", "label_win", "target", "label"]
                     if c in ticker_df.columns]
        X = ticker_df.drop(columns=meta_cols, errors="ignore").select_dtypes(include=["number"])
    else:
        X = ticker_df[available]

    prob = float(clf.predict_proba(X)[0][1])
    signal = 1 if prob >= 0.5 else 0

    return {
        "ticker": ticker_plain,
        "signal": signal,
        "probability": round(prob, 4),
        "model": ptr.get("model_file", "unknown").replace(".joblib", ""),
        "auc_wf": ptr.get("auc_wf", 0),
        "as_of": as_of,
    }


@router.get("/signal/{ticker}", summary="ML signal for a ticker")
async def get_signal(ticker: str):
    """
    XGBoost walk-forward signal.
    - signal: 1=buy, 0=no signal
    - probability: model confidence (0-1)
    - auc_wf: walk-forward AUC of current model
    - model: active model filename
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


@router.get("/model/info", summary="Current model metadata")
async def get_model_info():
    """Returns metadata of the currently active ML model (from current_model.json)."""
    _, ptr = _load_v6()
    return api_response(True, data=ptr)
