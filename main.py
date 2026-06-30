from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime, time
import pytz

from app.database import init_db, get_db
from app.routers import screen, analyze, backtest, broker_flow, paper_trade, config_router, debate, journal, portfolio as portfolio_router, ml_signal, ml_paper
from app.utils import api_response
from app.services.morning_report import generate_morning_report
from app.services.data_fetcher import fetch_universe_data, market_status
from app.services.data_quality_service import get_data_health, compute_ara_arb, validate_ohlcv
from app.services.risk_service import compute_atr_position_sizing, compute_single_stock_var, check_drawdown_circuit_breaker
from app.services.outcome_tracker_service import fill_pending_outcomes
from app.services.ml_paper_service import MLPaperTrade, fill_ml_outcomes  # registers model to Base

# Timezone setup
WIB = pytz.timezone("Asia/Jakarta")

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    app.state.db = await get_db().__anext__()
    print("[R12] swandy-fund API ready")
    ml_signal._load_v6()  # preload xgb_v6 at startup
    
    # Initialize scheduler
    scheduler = AsyncIOScheduler(timezone=WIB)
    
    # Add jobs
    # Outcome tracker T+3 — 16:30 WIB weekdays
    scheduler.add_job(
        fill_ml_outcomes,
        trigger=CronTrigger(hour=16, minute=35, day_of_week="mon-fri", timezone=WIB),
        id="ml_outcome_tracker",
        name="Fill ML paper trade T+3/SL outcomes",
        replace_existing=True,
    )

    scheduler.add_job(
        fill_pending_outcomes,
        trigger=CronTrigger(hour=16, minute=30, day_of_week="mon-fri", timezone=WIB),
        id="outcome_tracker_t3",
        name="Backfill T+3 outcomes",
        replace_existing=True
    )

    scheduler.add_job(
        generate_morning_report,
        args=[app.state.db],
        trigger=CronTrigger(hour=7, minute=30, timezone=WIB),
        id="morning_report",
        name="Generate morning market report",
        replace_existing=True
    )
    
    scheduler.add_job(
        fetch_universe_data,
        args=["lq45", app.state.db],
        trigger=CronTrigger(hour=9, minute=5, timezone=WIB),
        id="fetch_ohlcv",
        name="Fetch OHLCV data for universe",
        replace_existing=True
    )
    
    scheduler.add_job(
        fetch_universe_data,
        args=["lq45", app.state.db],
        trigger=CronTrigger(hour=10, minute=0, timezone=WIB),
        id="fetch_ohlcv_10am",
        name="Fetch OHLCV data for universe",
        replace_existing=True
    )
    
    scheduler.add_job(
        fetch_universe_data,
        args=["lq45", app.state.db],
        trigger=CronTrigger(hour=15, minute=30, timezone=WIB),
        id="fetch_ohlcv_eod",
        name="Fetch OHLCV data for universe",
        replace_existing=True
    )
    
    scheduler.start()
    
    yield
    
    scheduler.shutdown()
    print("[R12] shutting down...")

app = FastAPI(title="Swandy Fund AI", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    import httpx
    from sqlalchemy import text
    checks = {}

    # Ollama
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get("http://172.17.0.1:11434/api/tags", timeout=3)
            models = [m["name"] for m in r.json().get("models", [])]
            checks["ollama"] = {"status": "ok", "models": models}
    except Exception:
        checks["ollama"] = {"status": "down"}

    # Postgres
    try:
        from app.database import engine
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        checks["postgres"] = {"status": "ok"}
    except Exception:
        checks["postgres"] = {"status": "down"}

    # Qdrant (optional)
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get("http://r12-qdrant:6333/collections", timeout=3)
            checks["qdrant"] = {"status": "ok" if r.status_code == 200 else "degraded"}
    except Exception:
        checks["qdrant"] = {"status": "not_installed"}

    # Agent0 (data collector - do NOT restart)
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get("http://localhost:8001/health", timeout=3)
            checks["agent0"] = {"status": "ok", "note": "data collector - do not touch"}
    except Exception:
        checks["agent0"] = {"status": "unreachable", "note": "may be normal if agent0 has no /health"}

    # XGBoost model
    try:
        from app.services.ml_service import _load_model
        checks["xgboost"] = {"status": "ok" if _load_model() is not None else "model_not_found"}
    except Exception:
        checks["xgboost"] = {"status": "error"}

    critical = ["postgres"]  # ollama = fallback gateway, not critical for 5A
    all_ok = all(checks.get(k, {}).get("status") == "ok" for k in critical)

    return api_response(True, data={
        "status": "ok" if all_ok else "degraded",
        "version": "1.3.0",
        "phase": "5E",
        "dependencies": checks,
        "ts": datetime.utcnow().isoformat()
    })

@app.get("/market-status")
async def get_market_status():
    status = market_status()
    return api_response(True, data={"status": status})


@app.get("/api/risk/{ticker}")
async def get_risk_params(
    ticker: str,
    price: float = 0.0,
    portfolio: float = 50000000.0,
    risk_pct: float = 0.01,
    current_value: float = 0.0,
    peak_value: float = 0.0
):
    """
    Phase 5D: ATR-based position sizing + VaR + circuit breaker.
    Query params:
      price        : current stock price (IDR). If 0, uses last close from OHLCV.
      portfolio    : total portfolio value in IDR (default 50jt).
      risk_pct     : max loss per trade as fraction of portfolio (default 0.01 = 1%).
      current_value: current portfolio value for circuit breaker check.
      peak_value   : peak portfolio value for circuit breaker check.
    """
    import asyncio
    loop = asyncio.get_event_loop()

    # If price not provided, try to get from OHLCV
    if price <= 0:
        try:
            from sqlalchemy import text
            async with AsyncSessionLocal() as db:
                q = await db.execute(
                    text("SELECT close FROM ohlcv_daily WHERE ticker=:t ORDER BY date DESC LIMIT 1"),
                    {"t": ticker if ticker.endswith(".JK") else f"{ticker}.JK"}
                )
                row = q.fetchone()
                if row:
                    price = float(row[0])
        except Exception:
            pass

    if price <= 0:
        return api_response(False, error=f"Price not available for {ticker}. Pass ?price=XXXX")

    # ATR position sizing
    sizing = await loop.run_in_executor(
        None,
        lambda: compute_atr_position_sizing(ticker, price, portfolio, risk_pct)
    )

    # Single stock VaR
    var_data = await loop.run_in_executor(
        None,
        lambda: compute_single_stock_var(ticker, sizing["position_value_idr"])
    )

    # Circuit breaker (only if values provided)
    breaker = None
    if current_value > 0 and peak_value > 0:
        breaker = check_drawdown_circuit_breaker(current_value, peak_value)

    return api_response(True, data={
        "ticker": ticker,
        "phase": "5E",
        "position_sizing": sizing,
        "var": var_data,
        "circuit_breaker": breaker
    })


@app.get("/api/data/health/{ticker}")
async def get_ticker_data_health(ticker: str, db = Depends(get_db)):
    """Phase 5E: Data quality check — yfinance validation, ARA/ARB, DB freshness."""
    health = await get_data_health(ticker, db_session=db)
    return api_response(True, data=health)


@app.get("/api/data/ara-arb/{ticker}")
async def get_ara_arb(
    ticker: str,
    board: str = "regular"
):
    """
    Phase 5E: ARA/ARB limits for IDX stock.
    Fetches today and yesterday close from yfinance to compute limits.
    board: "regular" | "acceleration"
    """
    try:
        import yfinance as yf
        ticker_jk = ticker if ticker.endswith(".JK") else f"{ticker}.JK"
        df = yf.Ticker(ticker_jk).history(period="5d", interval="1d")
        if len(df) < 2:
            return api_response(False, error="Insufficient data for ARA/ARB calculation")
        prev_close = float(df.iloc[-2]["Close"])
        cur_close = float(df.iloc[-1]["Close"])
        result = compute_ara_arb(cur_close, prev_close, board=board)
        return api_response(True, data=result)
    except Exception as e:
        return api_response(False, error=str(e))

# Include all routers
app.include_router(screen.router)
app.include_router(analyze.router)
app.include_router(backtest.router)
app.include_router(broker_flow.router)
app.include_router(paper_trade.router)
app.include_router(config_router.router)
app.include_router(debate.router)
app.include_router(journal.router)
app.include_router(portfolio_router.router)

app.include_router(ml_signal.router)
app.include_router(ml_paper.router)

# db initialized in lifespan above