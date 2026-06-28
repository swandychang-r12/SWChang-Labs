import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import AIAnalysis
from .base_agent import BaseAgent
from .bull_analyst import BullAnalystAgent
from .bear_analyst import BearAnalystAgent
from .technical_analyst import TechnicalAnalystAgent
from .risk_analyst import RiskAnalystAgent
from .orchestrator import OrchestratorAgent
from app.services.ml_service import get_ml_probability
from app.services.qdrant_service import search_similar_setups, store_verdict
from app.services.similar_context_service import get_similar_context_prompt
from app.services.risk_service import compute_atr_position_sizing, compute_single_stock_var


async def _compute_rag_context(ticker: str) -> str:
    """
    Compute 83-dim features for current ticker (last trading day)
    and query Qdrant swandy_setups for similar past setups.
    Non-blocking — returns empty string on any failure.
    """
    try:
        import asyncio as _asyncio
        import sys as _sys
        _sys.path.insert(0, "/home/aiops/swandy-fund")
        from scripts.backfill_qdrant import compute_features, normalize_df, IDX_LIQUID_UNIVERSE
        import yfinance as _yf
        import pandas as _pd
        from datetime import datetime as _dt, timedelta as _td
        import numpy as _np

        # Fetch IHSG + ticker data for last 120 days (need enough for MA200)
        _end = _dt.now()
        _start = _end - _td(days=120)
        _yf_t = ticker + '.JK'
        _yf_ih = '^JKSE'

        # Download in executor to not block event loop
        loop = _asyncio.get_event_loop()
        def _dl():
            df_t  = _yf.download(_yf_t, start=_start, end=_end, interval='1d', progress=False, auto_adjust=True)
            df_ih = _yf.download(_yf_ih, start=_start, end=_end, interval='1d', progress=False, auto_adjust=True)
            return normalize_df(df_t), normalize_df(df_ih)

        df_t, df_ih = await loop.run_in_executor(None, _dl)

        if df_t.empty or len(df_t) < 20:
            return ""

        feat_df = compute_features(df_t, df_ih)
        if feat_df.empty:
            return ""

        # Use LAST row as current features dict
        last = feat_df.iloc[-1].to_dict()
        return await get_similar_context_prompt(ticker, last, top_k=3)

    except Exception as _e:
        import logging as _log
        _log.getLogger("debate_engine").warning(f"RAG context failed (non-fatal): {_e}")
        return ""


class DebateEngine:
    @staticmethod
    async def run_debate(
        ticker: str,
        market_data: Dict[str, Any],
        db_session: AsyncSession,
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        # Check cache first
        if not force_refresh:
            cached_result = await DebateEngine._get_cached_result(ticker, db_session)
            if cached_result:
                return cached_result

        # Initialize agents
        import yaml as _yaml, os as _os
        _cfg_path = _os.environ.get("CONFIG_YAML_PATH", "/app/config.yaml")
        with open(_cfg_path) as _f:
            _cfg = _yaml.safe_load(_f)
        _ac      = _cfg.get("agents", {})
        _gateway = _ac.get("ai_gateway", "http://172.17.0.1:20128/v1")
        _api_key = _ac.get("ai_api_key", "sk-23da001d912f62cd-z8qp0d-f192f4a3")

        def _kw(name, dt, dtok):
            a = _ac.get(name, {})
            return {"model": a.get("model", "groq/llama-3.3-70b-versatile"),
                    "temperature": a.get("temperature", dt),
                    "max_tokens": a.get("max_tokens", dtok),
                    "gateway_url": _gateway, "api_key": _api_key}

        bull_agent = BullAnalystAgent(
            name="bull_analyst", role="Bullish Analyst", **_kw("bull_analyst", 0.7, 800))
        bear_agent = BearAnalystAgent(
            name="bear_analyst", role="Bearish Analyst", **_kw("bear_analyst", 0.3, 800))
        technical_agent = TechnicalAnalystAgent(
            name="technical_analyst", role="Technical Analyst", **_kw("technical_analyst", 0.4, 600))
        risk_agent = RiskAnalystAgent(
            name="risk_analyst", role="Risk Analyst", **_kw("risk_analyst", 0.2, 500))
        orchestrator = OrchestratorAgent(
            name="orchestrator", role="Chief Investment Officer", **_kw("orchestrator", 0.1, 1200))

        # Run 4 agents + ML signal + Qdrant similarity concurrently
        start_time = time.monotonic()
        (reports, ml_signal, similar_setups, rag_ctx) = await asyncio.gather(
            asyncio.gather(
                bull_agent.analyze(ticker, market_data),
                bear_agent.analyze(ticker, market_data),
                technical_agent.analyze(ticker, market_data),
                risk_agent.analyze(ticker, market_data),
                return_exceptions=True
            ),
            get_ml_probability(ticker),
            search_similar_setups(ticker, limit=3),
            _compute_rag_context(ticker),
        )

        # Build ml_context
        if ml_signal.get("available"):
            ml_context = (
                f"XGBoost v5 probability: {ml_signal['probability']:.1%} "
                f"({ml_signal['signal']}). "
                f"Ini signal yang backtested dan terkalibrasi - jadikan anchor utama jika LLM confidence lemah."
            )
        else:
            ml_context = "XGBoost signal tidak tersedia untuk ticker ini."

        # Build similar setups context
        if similar_setups:
            setups_text = "; ".join(
                f"{s['ticker']}@{s['date']}: {s['action']} (ML:{s['ml_probability']:.1%}, sim:{s['similarity']:.2f})"
                for s in similar_setups
            )
            similar_context = f"Setup serupa di masa lalu: {setups_text}"
        else:
            similar_context = "Belum ada setup serupa di database (akan terakumulasi seiring waktu)."

        # Merge 83-dim RAG context if available
        if rag_ctx:
            similar_context = rag_ctx  # 83-dim vector RAG supersedes basic search
        elif not similar_setups:
            similar_context = "Belum ada setup serupa di database (akan terakumulasi seiring waktu)."

        # Prepare data for orchestrator
        orchestrator_data = {
            "bull": reports[0] if not isinstance(reports[0], Exception) else {"error": str(reports[0])},
            "bear": reports[1] if not isinstance(reports[1], Exception) else {"error": str(reports[1])},
            "technical": reports[2] if not isinstance(reports[2], Exception) else {"error": str(reports[2])},
            "risk": reports[3] if not isinstance(reports[3], Exception) else {"error": str(reports[3])},
            "ml_signal": ml_signal,
            "ml_context": ml_context,
            "similar_context": similar_context
        }

        # Get final verdict
        verdict = await orchestrator.analyze(ticker, orchestrator_data)

        # Prepare result
        elapsed_total_s = time.monotonic() - start_time
        result = {
            "ticker": ticker,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "elapsed_total_s": elapsed_total_s,
            "agents": {
                "bull": reports[0] if not isinstance(reports[0], Exception) else {"error": str(reports[0])},
                "bear": reports[1] if not isinstance(reports[1], Exception) else {"error": str(reports[1])},
                "technical": reports[2] if not isinstance(reports[2], Exception) else {"error": str(reports[2])},
                "risk": reports[3] if not isinstance(reports[3], Exception) else {"error": str(reports[3])}
            },
            "ml_signal": ml_signal,
            "similar_setups": similar_setups,
            "verdict": verdict
        }

        # Cache result in Postgres
        await DebateEngine._cache_result(result, db_session)

        # Store verdict in Qdrant (background, non-critical)
        try:
            await store_verdict(
                ticker=ticker,
                date_str=result["date"],
                action=verdict.get("action", "HOLD"),
                confidence=float(verdict.get("confidence", 0.5)),
                ml_probability=float(ml_signal.get("probability", 0.5)),
                verdict=verdict
            )
        except Exception:
            pass

        # Phase 5D: compute risk params from last close in OHLCV
        risk_params = None
        try:
            from sqlalchemy import text
            async with db_session.__class__() as _db2:
                pass  # skip; use features parquet price
        except Exception:
            pass
        try:
            from app.services.risk_service import compute_atr_position_sizing, compute_single_stock_var
            import asyncio as _asyncio
            _features = await _asyncio.get_event_loop().run_in_executor(
                None,
                lambda: __import__('app.services.risk_service', fromlist=['get_ticker_features']).get_ticker_features(ticker)
            )
            if _features:
                # Use atr_ratio as proxy for price (atr / atr_ratio * 100)
                _atr = _features.get('atr_14', 0)
                _atr_ratio = _features.get('atr_ratio', 0)
                if _atr > 0 and _atr_ratio > 0:
                    _price = _atr / (_atr_ratio / 100.0)
                    _sizing = await _asyncio.get_event_loop().run_in_executor(
                        None, lambda: compute_atr_position_sizing(ticker, _price)
                    )
                    _var = await _asyncio.get_event_loop().run_in_executor(
                        None, lambda: compute_single_stock_var(ticker, _sizing["position_value_idr"])
                    )
                    risk_params = {"position_sizing": _sizing, "var": _var}
        except Exception as _re:
            pass
        result["risk_params"] = risk_params

        return result

    @staticmethod
    async def _get_cached_result(ticker: str, db_session: AsyncSession) -> Optional[Dict[str, Any]]:
        four_hours_ago = datetime.now() - timedelta(hours=4)
        stmt = select(AIAnalysis).where(
            AIAnalysis.ticker == ticker,
            AIAnalysis.created_at >= four_hours_ago
        ).order_by(AIAnalysis.created_at.desc())
        result = await db_session.execute(stmt)
        analysis = result.scalars().first()

        if analysis:
            return {
                "ticker": analysis.ticker,
                "date": analysis.analysis_date.strftime("%Y-%m-%d"),
                "cached": True,
                "verdict": analysis.debate_json.get("verdict", {})
            }
        return None

    @staticmethod
    async def _cache_result(result: Dict[str, Any], db_session: AsyncSession) -> None:
        analysis_date = datetime.strptime(result["date"], "%Y-%m-%d").date()
        debate_json = {
            "agents": result["agents"],
            "verdict": result["verdict"],
            "ml_signal": result.get("ml_signal", {}),
            "similar_setups": result.get("similar_setups", [])
        }

        new_analysis = AIAnalysis(
            ticker=result["ticker"],
            analysis_date=analysis_date,
            debate_json=debate_json,
            action=result["verdict"].get("action", "HOLD"),
            confidence=float(result["verdict"].get("confidence", 0.5))
        )

        db_session.add(new_analysis)
        await db_session.commit()
