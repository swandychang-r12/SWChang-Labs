"""
SERVICE-SIMILAR-CONTEXT-20260628
Sprint 5F — Task 4: Query Qdrant for similar past setups
Used by orchestrator to inject historical context into LLM prompt.

Usage (di orchestrator_service.py atau debate router):
    from app.services.similar_context_service import get_similar_context_prompt
    ctx = await get_similar_context_prompt(ticker, features_dict, top_k=3)
    # inject ctx into orchestrator system prompt
"""
import os
import logging
from typing import Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

log = logging.getLogger(__name__)

QDRANT_HOST = os.getenv('QDRANT_HOST', 'localhost')
QDRANT_PORT = int(os.getenv('QDRANT_PORT', 6333))
COLLECTION  = 'swandy_setups'
VECTOR_DIM  = 83

# Feature column order must match backfill_qdrant.py exactly
FEATURE_COLS = [
    # Group A (13)
    'rs_vs_ihsg_1d','rs_vs_ihsg_5d','rs_vs_ihsg_10d','rs_vs_ihsg_20d',
    'rs_rank_universe','price_return_1d','price_return_3d','price_return_5d',
    'price_return_8d','price_return_20d','roc_5','roc_10','momentum_accel',
    # Group B (13)
    'vol_ratio_5d','vol_ratio_10d','vol_ratio_20d','idr_volume','idr_volume_log',
    'vol_trend_5d','vol_up_days_ratio','obv_slope_5d','obv_divergence',
    'vwap_deviation','price_vol_corr_5d','vol_anomaly_flag','avg_vol_idr_20d',
    # Group C (20)
    'rsi_14','rsi_7','rsi_zone','macd_line','macd_signal','macd_histogram',
    'macd_crossover','macd_hist_slope','stoch_k','stoch_d','stoch_crossover',
    'ema5_position','ema10_position','ema20_position','ema50_position','ema200_position',
    'ma_stack','ema20_ema50_cross','cci_14','williams_r',
    # Group D (13)
    'atr_14','atr_ratio','atr_trend','bb_upper','bb_lower','bb_pct_b',
    'bb_width','bb_squeeze_flag','w52_position','hist_vol_5d','hist_vol_20d',
    'vol_ratio_hvol','consolidation_score',
    # Group E (8)
    'ihsg_ma20_regime','ihsg_ma50_regime','ihsg_ma200_regime','ihsg_regime_score',
    'ihsg_return_1d','ihsg_return_5d','ihsg_vol_ratio','stock_beta_ihsg',
    # Group F (7)
    'foreign_flow_proxy','foreign_accumulation_3d','corp_action_proximity',
    'corp_action_flag','sector_rs_vs_ihsg_5d','sector_rank','stock_vs_sector_5d',
    # Group G (5)
    'vol_rs_combo','rsi_vol_signal','ma_stack_with_volume',
    'ihsg_filtered_score','setup_quality_score',
    # Extra (4)
    'close_norm','high_norm','low_norm','volume_norm',
]

assert len(FEATURE_COLS) == VECTOR_DIM, \
    f"FEATURE_COLS has {len(FEATURE_COLS)} cols, expected {VECTOR_DIM}"


def _build_vector_from_dict(features: dict) -> list[float]:
    """Build 83-dim vector from features dict. Missing keys → 0.0."""
    vec = [float(features.get(k, 0.0)) for k in FEATURE_COLS]
    vec = np.array(vec, dtype=float)
    vec = np.nan_to_num(vec, nan=0.0, posinf=5.0, neginf=-5.0)
    vec = np.clip(vec, -10.0, 10.0)
    return vec.tolist()


def _client() -> QdrantClient:
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=10)


def get_similar_setups(
    ticker: str,
    features: dict,
    top_k: int = 5,
    min_score: float = 0.70,
    filter_same_ticker: bool = False,
    only_labeled: bool = False,
) -> list[dict]:
    """
    Query Qdrant for top_k most similar past setups.

    Args:
        ticker:              current ticker being analyzed
        features:            dict of feature_name→float (83 keys from FEATURE_COLS)
        top_k:               number of similar setups to return
        min_score:           minimum cosine similarity (0.0–1.0)
        filter_same_ticker:  if True, restrict to same ticker only
        only_labeled:        if True, only return setups with known outcome

    Returns:
        list of dicts with keys: ticker, date, score, payload
    """
    try:
        client = _client()
    except Exception as e:
        log.warning(f"Qdrant connection failed: {e}")
        return []

    # Check collection exists
    try:
        cols = [c.name for c in client.get_collections().collections]
        if COLLECTION not in cols:
            log.warning(f"Collection '{COLLECTION}' not found in Qdrant")
            return []
        info = client.get_collection(COLLECTION)
        if info.points_count == 0:
            log.info("Qdrant collection is empty — no similar context available")
            return []
    except Exception as e:
        log.warning(f"Qdrant collection check failed: {e}")
        return []

    # Build query vector
    query_vec = _build_vector_from_dict(features)

    # Optional filters
    must_conditions = []
    if filter_same_ticker:
        must_conditions.append(
            FieldCondition(key='ticker', match=MatchValue(value=ticker))
        )
    if only_labeled:
        must_conditions.append(
            FieldCondition(key='label_win', match=MatchValue(value=True))
        )

    q_filter = Filter(must=must_conditions) if must_conditions else None

    try:
        result = client.query_points(
            collection_name=COLLECTION,
            query=query_vec,
            limit=top_k,
            score_threshold=min_score,
            query_filter=q_filter,
            with_payload=True,
        )
        results = result.points
    except Exception as e:
        log.warning(f"Qdrant search failed: {e}")
        return []

    similar = []
    for hit in results:
        p = hit.payload or {}
        similar.append({
            'ticker':              p.get('ticker', '?'),
            'date':                p.get('date', '?'),
            'similarity_score':    round(hit.score, 3),
            'setup_quality':       round(p.get('setup_quality_score', 0), 1),
            'rsi_14':              round(p.get('rsi_14', 0), 1),
            'vol_ratio_20d':       round(p.get('vol_ratio_20d', 0), 2),
            'rs_vs_ihsg_5d':       round(p.get('rs_vs_ihsg_5d', 0) * 100, 2),
            'ihsg_regime_score':   p.get('ihsg_regime_score', 0),
            'ma_stack':            p.get('ma_stack', False),
            'label_win':           p.get('label_win'),     # None if not yet known
            'outcome_filled':      p.get('outcome_filled', False),
        })

    return similar


def format_similar_context_for_prompt(
    similar: list[dict],
    ticker: str,
) -> str:
    """
    Format similar setups list into a concise text block for LLM injection.
    Keeps it short — max ~200 tokens for orchestrator context window budget.
    """
    if not similar:
        return ""

    lines = [
        f"## Similar Past Setups for {ticker} (Qdrant RAG — top {len(similar)} matches)",
        "",
    ]

    for i, s in enumerate(similar, 1):
        outcome_str = '?'
        if s['label_win'] is True:
            outcome_str = '✅ WIN (+2%+ in 3d)'
        elif s['label_win'] is False:
            outcome_str = '❌ LOSS'

        lines.append(
            f"{i}. {s['ticker']} {s['date']} "
            f"[sim={s['similarity_score']:.2f}] "
            f"QS={s['setup_quality']:.0f} "
            f"RSI={s['rsi_14']:.0f} "
            f"VolR={s['vol_ratio_20d']:.1f}x "
            f"RS={s['rs_vs_ihsg_5d']:+.1f}% "
            f"Regime={int(s['ihsg_regime_score'])}/3 "
            f"→ {outcome_str}"
        )

    lines += [
        "",
        "Use these historical analogs to inform your confidence assessment.",
        "High-similarity WIN cases increase conviction. LOSS cases require extra caution.",
    ]
    return "\n".join(lines)


async def get_similar_context_prompt(
    ticker: str,
    features: dict,
    top_k: int = 3,
) -> str:
    """
    Async wrapper — fetch similar setups and return formatted prompt string.
    Returns empty string if Qdrant unavailable (non-blocking).
    """
    try:
        similar = get_similar_setups(
            ticker=ticker,
            features=features,
            top_k=top_k,
            min_score=0.72,
            filter_same_ticker=False,
            only_labeled=False,
        )
        return format_similar_context_for_prompt(similar, ticker)
    except Exception as e:
        log.warning(f"similar_context_service error (non-fatal): {e}")
        return ""
