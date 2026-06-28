from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime, time
import pytz

from app.database import init_db, get_db
from app.routers import screen, analyze, backtest, broker_flow, paper_trade, config_router, debate, journal
from app.utils import api_response
from app.services.morning_report import generate_morning_report
from app.services.data_fetcher import fetch_universe_data, market_status

# Timezone setup
WIB = pytz.timezone("Asia/Jakarta")

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    app.state.db = await get_db().__anext__()
    print("[R12] swandy-fund API ready")
    
    # Initialize scheduler
    scheduler = AsyncIOScheduler(timezone=WIB)
    
    # Add jobs
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
    return api_response(True, data={"status": "ok", "version": "1.0.0"})

@app.get("/market-status")
async def get_market_status():
    status = market_status()
    return api_response(True, data={"status": status})

# Include all routers
app.include_router(screen.router)
app.include_router(analyze.router)
app.include_router(backtest.router)
app.include_router(broker_flow.router)
app.include_router(paper_trade.router)
app.include_router(config_router.router)
app.include_router(debate.router)
app.include_router(journal.router)

# db initialized in lifespan above
