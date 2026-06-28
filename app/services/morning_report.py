from datetime import datetime, date
import yfinance as yf
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import MorningReport
from app.agents.orchestrator import OrchestratorAgent
from app.services.data_fetcher import fetch_universe_data, UNIVERSE

async def generate_morning_report(db_session: AsyncSession) -> str:
    """Generate AI-powered morning market report"""
    today = date.today()
    stmt = select(MorningReport).where(MorningReport.date == today)
    result = await db_session.execute(stmt)
    existing_report = result.scalars().first()
    if existing_report:
        return existing_report.content
    try:
        ihsg = yf.Ticker("^JKSE")
        ihsg_hist = ihsg.history(period="2d")
        if not ihsg_hist.empty:
            ihsg_last = ihsg_hist.iloc[-1]["Close"]
            ihsg_prev = ihsg_hist.iloc[-2]["Close"] if len(ihsg_hist) > 1 else ihsg_last
            ihsg_change = (ihsg_last - ihsg_prev) / ihsg_prev * 100
        else:
            ihsg_last = 7000
            ihsg_change = 0.0
        top_stocks = await fetch_universe_data("lq45", db_session)
        top_5 = sorted(top_stocks[:10], key=lambda x: x["data"]["change_pct"], reverse=True)[:5]
        dow = yf.Ticker("^DJI").history(period="1d")
        nasdaq = yf.Ticker("^IXIC").history(period="1d")
        nikkei = yf.Ticker("^N225").history(period="1d")
        dow_change = (dow.iloc[-1]["Close"] - dow.iloc[-1]["Open"]) / dow.iloc[-1]["Open"] * 100 if not dow.empty else 0.0
        nasdaq_change = (nasdaq.iloc[-1]["Close"] - nasdaq.iloc[-1]["Open"]) / nasdaq.iloc[-1]["Open"] * 100 if not nasdaq.empty else 0.0
        nikkei_change = (nikkei.iloc[-1]["Close"] - nikkei.iloc[-1]["Open"]) / nikkei.iloc[-1]["Open"] * 100 if not nikkei.empty else 0.0
    except Exception as e:
        print(f"Error fetching market data: {e}")
        ihsg_last = 7000
        ihsg_change = 0.0
        top_5 = []
        dow_change = 0.0
        nasdaq_change = 0.0
        nikkei_change = 0.0
    context = {
        "ihsg": {"level": round(ihsg_last, 2), "change_pct": round(ihsg_change, 2)},
        "top_5_stocks": [{"ticker": stock["ticker"], "price": stock["data"]["close"], "change_pct": stock["data"]["change_pct"]} for stock in top_5],
        "global_markets": {"dow": round(dow_change, 2), "nasdaq": round(nasdaq_change, 2), "nikkei": round(nikkei_change, 2)},
        "date": today.isoformat()
    }
    orchestrator = OrchestratorAgent(
        name="morning_reporter",
        role="Market Analyst",
        model="qwen2.5:7b",
        temperature=0.7,
        max_tokens=2048,
        gateway_url="http://localhost:11434/v1",
        api_key="ollama"
    )
    report = await orchestrator.analyze(
        ticker="IHSG",
        market_data={"context": context, "task": "Generate morning market report in Bahasa Indonesia (300-400 words)"}
    )
    report_content = report.get("reasoning", "No report generated")
    new_report = MorningReport(date=today, content=report_content, meta_data=context)
    db_session.add(new_report)
    await db_session.commit()
    return report_content

async def get_latest_morning_report(db_session: AsyncSession) -> dict:
    """Get the latest morning report"""
    stmt = select(MorningReport).order_by(MorningReport.date.desc()).limit(1)
    result = await db_session.execute(stmt)
    report = result.scalars().first()
    if report:
        return {"date": report.date.isoformat(), "content": report.content, "meta_data": report.meta_data, "created_at": report.created_at.isoformat()}
    return None
