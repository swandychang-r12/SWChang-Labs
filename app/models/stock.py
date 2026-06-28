from sqlalchemy import Column, String, Date, Numeric, BigInteger, TIMESTAMP, Integer
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from app.database import Base

class OHLCVCache(Base):
    __tablename__ = "ohlcv_cache"
    ticker   = Column(String(12), primary_key=True)
    date     = Column(Date,       primary_key=True)
    interval = Column(String(5),  primary_key=True)
    open     = Column(Numeric(15, 2))
    high     = Column(Numeric(15, 2))
    low      = Column(Numeric(15, 2))
    close    = Column(Numeric(15, 2))
    volume   = Column(BigInteger)
    adj_close = Column(Numeric(15, 2))
    cached_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

class ScreenerResult(Base):
    __tablename__ = "screener_results"
    id           = Column(Integer, primary_key=True, autoincrement=True)
    ticker       = Column(String(12), nullable=False)
    scan_date    = Column(Date, nullable=False)
    ml_score     = Column(Numeric(6, 4))
    volume_ratio = Column(Numeric(8, 2))
    price        = Column(Numeric(15, 2))
    change_pct   = Column(Numeric(8, 4))
    action       = Column(String(12))
    indicators   = Column(JSONB)
    created_at   = Column(TIMESTAMP(timezone=True), server_default=func.now())
