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
        _gateway = _ac.get("ai_gateway", "http://localhost:20128/v1")
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

        # Run agents in parallel
        start_time = time.monotonic()
        reports = await asyncio.gather(
            bull_agent.analyze(ticker, market_data),
            bear_agent.analyze(ticker, market_data),
            technical_agent.analyze(ticker, market_data),
            risk_agent.analyze(ticker, market_data),
            return_exceptions=True
        )

        # Prepare data for orchestrator
        orchestrator_data = {
            "bull": reports[0] if not isinstance(reports[0], Exception) else {"error": str(reports[0])},
            "bear": reports[1] if not isinstance(reports[1], Exception) else {"error": str(reports[1])},
            "technical": reports[2] if not isinstance(reports[2], Exception) else {"error": str(reports[2])},
            "risk": reports[3] if not isinstance(reports[3], Exception) else {"error": str(reports[3])}
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
            "verdict": verdict
        }

        # Cache the result
        await DebateEngine._cache_result(result, db_session)

        return result

    @staticmethod
    async def _get_cached_result(ticker: str, db_session: AsyncSession) -> Optional[Dict[str, Any]]:
        # Check if there's a recent analysis for this ticker
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
        # Save the result to database
        analysis_date = datetime.strptime(result["date"], "%Y-%m-%d").date()
        debate_json = {
            "agents": result["agents"],
            "verdict": result["verdict"]
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
