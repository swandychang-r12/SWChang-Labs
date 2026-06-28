import sqlite3, asyncio
from functools import partial
from typing import Dict, Any, List

BEI_DB = "/trading-scanner/data/bei_foreign.db"
STOCKBIT_DIR = "/trading-scanner/data/stockbit/"

def _query_bei_sync(ticker: str, days: int) -> Dict[str, Any]:
    """Read broker flow directly from SQLite (no import of bei_scraper needed)."""
    try:
        conn = sqlite3.connect(BEI_DB)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        # Try common table names
        tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        if not tables:
            conn.close()
            return {"error": "No tables in bei_foreign.db", "ticker": ticker}
        # Use first matching table
        table = tables[0]
        try:
            rows = cur.execute(
                f"SELECT * FROM {table} WHERE ticker=? ORDER BY date DESC LIMIT ?",
                (ticker, days)
            ).fetchall()
        except Exception:
            rows = cur.execute(
                f"SELECT * FROM {table} ORDER BY date DESC LIMIT ?", (days,)
            ).fetchall()
        conn.close()
        return {
            "ticker": ticker,
            "table": table,
            "rows": [dict(r) for r in rows],
            "count": len(rows),
        }
    except Exception as e:
        return {"error": str(e), "ticker": ticker}

async def get_broker_flow(ticker: str, days: int = 5) -> Dict[str, Any]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, partial(_query_bei_sync, ticker, days))

def _read_paper_state_sync() -> Dict[str, Any]:
    import json, os
    state_path = "/trading-scanner/data/paper_state.json"
    if not os.path.exists(state_path):
        return {"error": "paper_state.json not found"}
    with open(state_path, "r", encoding="utf-8") as f:
        return json.load(f)

async def get_paper_state() -> Dict[str, Any]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _read_paper_state_sync)
