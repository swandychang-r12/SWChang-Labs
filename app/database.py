from datetime import datetime, date
from sqlalchemy import Column, Integer, String, Date, Numeric, TIMESTAMP, Float, Text, Index, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from app.config import settings

engine = create_async_engine(
    settings.database_url,
    echo=False,
    pool_size=10,
    max_overflow=5,
)

AsyncSessionLocal = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

Base = declarative_base()

class AIAnalysis(Base):
    __tablename__ = "ai_analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String(12), nullable=False, index=True)
    analysis_date = Column(Date, nullable=False, index=True)
    debate_json = Column(JSONB, nullable=False)
    action = Column(String(15), nullable=False)
    confidence = Column(Numeric(5, 4), nullable=False)
    created_at = Column(TIMESTAMP, default=func.now(), nullable=False)

class OHLCVDaily(Base):
    __tablename__ = "ohlcv_daily"
    __table_args__ = (Index('ix_ohlcv_daily_ticker_date', 'ticker', 'date'),)
    
    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String(12), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False)
    adjusted_close = Column(Float, nullable=True)
    created_at = Column(TIMESTAMP, default=func.now(), nullable=False)

class OHLCV5m(Base):
    __tablename__ = "ohlcv_5m"
    __table_args__ = (Index('ix_ohlcv_5m_ticker_datetime', 'ticker', 'datetime'),)
    
    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String(12), nullable=False, index=True)
    datetime = Column(TIMESTAMP, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False)
    created_at = Column(TIMESTAMP, default=func.now(), nullable=False)

class JournalTrade(Base):
    __tablename__ = "journal_trades"
    
    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String(12), nullable=False, index=True)
    side = Column(String(4), nullable=False)  # BUY or SELL
    entry_price = Column(Float, nullable=False)
    stop_loss = Column(Float, nullable=True)
    target_price = Column(Float, nullable=True)
    lot_size = Column(Integer, nullable=False)
    notes = Column(Text, nullable=True)
    strategy_used = Column(String(50), nullable=True)
    signal_date = Column(Date, nullable=False)
    exit_price = Column(Float, nullable=True)
    exit_date = Column(Date, nullable=True)
    pnl_idr = Column(Float, nullable=True)
    pnl_pct = Column(Float, nullable=True)
    created_at = Column(TIMESTAMP, default=func.now(), nullable=False)
    updated_at = Column(TIMESTAMP, default=func.now(), onupdate=func.now(), nullable=False)

class PostTradeReview(Base):
    __tablename__ = "post_trade_reviews"
    
    id = Column(Integer, primary_key=True, index=True)
    trade_id = Column(Integer, nullable=False, index=True)  # FK to journal_trades
    what_happened = Column(Text, nullable=False)
    what_worked = Column(Text, nullable=True)
    what_to_change = Column(Text, nullable=True)
    emotion_score = Column(Integer, nullable=True)  # 1-5 scale
    ai_summary = Column(Text, nullable=True)
    created_at = Column(TIMESTAMP, default=func.now(), nullable=False)

class MorningReport(Base):
    __tablename__ = "morning_reports"
    
    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, nullable=False, unique=True, index=True)
    content = Column(Text, nullable=False)
    meta_data = Column(JSONB, nullable=True)  # renamed from metadata to avoid conflict
    created_at = Column(TIMESTAMP, default=func.now(), nullable=False)

async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("[DB] Tables created / verified")
