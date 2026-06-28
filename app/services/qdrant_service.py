"""
qdrant_service.py — Phase 5C: Similar setup search via Qdrant
Collection: swandy_setups
Vector: XGBoost 83-dim feature vector (already normalized from backtest)
Payload: ticker, date, verdict, ml_probability, action, confidence
"""
import asyncio
import hashlib
import logging
from datetime import datetime
from typing import Optional
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

QDRANT_BASE = "http://r12-qdrant:6333"
COLLECTION = "swandy_setups"
VECTOR_SIZE = 83

# Feature order MUST match XGBoost model features
FEATURE_NAMES = [
    'A01_rs_vs_ihsg_1d', 'A02_rs_vs_ihsg_5d', 'A03_rs_vs_ihsg_10d', 'A04_rs_vs_ihsg_20d',
    'A05_rs_rank_universe', 'A06_price_return_1d', 'A07_price_return_3d', 'A08_price_return_5d',
    'A09_price_return_8d', 'A10_price_return_20d', 'A11_roc_5', 'A12_roc_10', 'A13_momentum_accel',
    'B01_vol_ratio_5d', 'B02_vol_ratio_10d', 'B03_vol_ratio_20d', 'B04_idr_volume',
    'B05_idr_volume_log', 'B06_vol_trend_5d', 'B07_vol_up_ratio', 'B08_obv_slope_5d',
    'B09_obv_divergence', 'B10_vwap_deviation', 'B11_pv_corr_5d', 'B12_vol_anomaly',
    'B13_avg_idr_20d', 'C01_rsi_14', 'C02_rsi_7', 'C03_rsi_zone', 'C04_macd_line',
    'C05_macd_signal', 'C06_macd_hist', 'C07_macd_cross', 'C08_macd_hist_slope',
    'C09_stoch_k', 'C10_stoch_d', 'C11_stoch_cross', 'C12_ema5_pos', 'C13_ema10_pos',
    'C14_ema20_pos', 'C15_ema50_pos', 'C16_ema200_pos', 'C17_ma_stack', 'C18_ema20_50',
    'C19_cci_14', 'C20_williams_r', 'D01_atr_14', 'D02_atr_ratio', 'D03_atr_trend',
    'D04_bb_upper', 'D05_bb_lower', 'D06_bb_pct_b', 'D07_bb_width', 'D08_bb_squeeze',
    'D09_w52_pos', 'D10_hvol_5d', 'D11_hvol_20d', 'D12_hvol_ratio', 'D13_consol',
    'E01_ihsg_ma20', 'E02_ihsg_ma50', 'E03_ihsg_ma200', 'E04_ihsg_regime', 'E05_ihsg_ret1d',
    'E06_ihsg_ret5d', 'E07_ihsg_vol_ratio', 'E08_beta_20d', 'E09_jkii_vs_ihsg_5d',
    'E10_jkii_ma20', 'F01_foreign_flow', 'F02_foreign_accum3d', 'F03_corp_prox',
    'F04_corp_flag', 'F05_sector_rs5d', 'F06_sector_rank', 'F07_stock_vs_sector',
    'F08_foreign_net_idr_log', 'F09_sec_vs_ihsg', 'G01_vol_rs_combo', 'G02_rsi_vol',
    'G03_mastack_vol', 'G04_regime_score', 'G05_setup_quality'
]


def _ticker_date_id(ticker: str, date_str: str) -> int:
    """Deterministic int ID from ticker+date."""
    h = hashlib.md5(f"{ticker}_{date_str}".encode()).hexdigest()
    return int(h[:8], 16)


async def ensure_collection() -> bool:
    """Create collection if not exists. Returns True if OK."""
    async with httpx.AsyncClient() as c:
        try:
            r = await c.get(f"{QDRANT_BASE}/collections/{COLLECTION}", timeout=5)
            if r.status_code == 200:
                return True
            # Create collection
            r2 = await c.put(
                f"{QDRANT_BASE}/collections/{COLLECTION}",
                json={
                    "vectors": {
                        "size": VECTOR_SIZE,
                        "distance": "Cosine"
                    }
                },
                timeout=10
            )
            return r2.status_code in (200, 201)
        except Exception as e:
            logger.error(f"Qdrant collection setup failed: {e}")
            return False


async def get_features_vector(ticker: str) -> Optional[list]:
    """Get XGBoost feature vector for ticker from features.parquet."""
    def _load():
        from pathlib import Path
        import pandas as pd
        from functools import lru_cache

        scanner = Path("/trading-scanner")
        fp = scanner / "data" / "features.parquet"
        if not fp.exists():
            return None

        df = pd.read_parquet(fp)
        ticker_jk = ticker if ticker.endswith(".JK") else f"{ticker}.JK"

        if "ticker" in df.columns:
            mask = df["ticker"].isin([ticker, ticker_jk])
        else:
            mask = df.index.isin([ticker, ticker_jk])

        row = df[mask].tail(1)
        if row.empty:
            return None

        # Extract features in canonical order
        vec = []
        for fname in FEATURE_NAMES:
            if fname in row.columns:
                val = float(row[fname].iloc[0])
                # Handle NaN
                if val != val:
                    val = 0.0
            else:
                val = 0.0
            vec.append(val)
        return vec

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _load)


async def store_verdict(
    ticker: str,
    date_str: str,
    action: str,
    confidence: float,
    ml_probability: float,
    verdict: dict
) -> bool:
    """Store debate verdict in Qdrant for future similarity search."""
    try:
        vec = await get_features_vector(ticker)
        if vec is None:
            logger.warning(f"No features for {ticker}, skip Qdrant store")
            return False

        if not await ensure_collection():
            return False

        point_id = _ticker_date_id(ticker, date_str)
        payload = {
            "ticker": ticker,
            "date": date_str,
            "action": action,
            "confidence": confidence,
            "ml_probability": ml_probability,
            "executive_summary": verdict.get("executive_summary", "")[:500],
            "stored_at": datetime.utcnow().isoformat()
        }

        async with httpx.AsyncClient() as c:
            r = await c.put(
                f"{QDRANT_BASE}/collections/{COLLECTION}/points",
                json={"points": [{"id": point_id, "vector": vec, "payload": payload}]},
                timeout=10
            )
            return r.status_code in (200, 201)
    except Exception as e:
        logger.error(f"Qdrant store failed for {ticker}: {e}")
        return False


async def search_similar_setups(ticker: str, limit: int = 3) -> list:
    """Find past similar setups for this ticker's current market features."""
    try:
        vec = await get_features_vector(ticker)
        if vec is None:
            return []

        if not await ensure_collection():
            return []

        async with httpx.AsyncClient() as c:
            r = await c.post(
                f"{QDRANT_BASE}/collections/{COLLECTION}/points/search",
                json={
                    "vector": vec,
                    "limit": limit + 1,  # +1 to exclude self
                    "with_payload": True,
                    "score_threshold": 0.85
                },
                timeout=10
            )
            if r.status_code != 200:
                return []

            results = r.json().get("result", [])
            # Exclude same ticker same date
            today = datetime.now().strftime("%Y-%m-%d")
            return [
                {
                    "ticker": p["payload"]["ticker"],
                    "date": p["payload"]["date"],
                    "action": p["payload"]["action"],
                    "confidence": p["payload"]["confidence"],
                    "ml_probability": p["payload"]["ml_probability"],
                    "similarity": round(p["score"], 3)
                }
                for p in results
                if not (p["payload"]["ticker"] == ticker and p["payload"]["date"] == today)
            ][:limit]
    except Exception as e:
        logger.error(f"Qdrant search failed for {ticker}: {e}")
        return []
