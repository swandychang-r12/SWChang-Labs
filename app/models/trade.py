from sqlalchemy import Column, String, Date, Numeric, Integer, TIMESTAMP, ARRAY, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from app.database import Base

class BacktestRun(Base):
    __tablename__ = "backtest_runs"
    id               = Column(Integer, primary_key=True, autoincrement=True)
    strategy         = Column(String(50))
    tickers          = Column(ARRAY(Text))
    start_date       = Column(Date)
    end_date         = Column(Date)
    initial_capital  = Column(Numeric(20, 2))
    final_capital    = Column(Numeric(20, 2))
    total_return_pct = Column(Numeric(10, 4))
    max_drawdown_pct = Column(Numeric(10, 4))
    sharpe_ratio     = Column(Numeric(8, 4))
    win_rate         = Column(Numeric(6, 4))
    total_trades     = Column(Integer)
    results          = Column(JSONB)
    created_at       = Column(TIMESTAMP(timezone=True), server_default=func.now())

class PaperTrade(Base):
    __tablename__ = "paper_trades_log"
    id          = Column(Integer, primary_key=True, autoincrement=True)
    ticker      = Column(String(12))
    side        = Column(String(4))
    entry_price = Column(Numeric(15, 2))
    entry_date  = Column(Date)
    exit_price  = Column(Numeric(15, 2), nullable=True)
    exit_date   = Column(Date, nullable=True)
    lot_size    = Column(Integer)
    pnl_idr    = Column(Numeric(20, 2), nullable=True)
    pnl_pct    = Column(Numeric(8, 4),  nullable=True)
    status      = Column(String(10), default="OPEN")
    strategy    = Column(String(50))
    created_at  = Column(TIMESTAMP(timezone=True), server_default=func.now())
