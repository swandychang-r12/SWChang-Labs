import sys, asyncio
from functools import partial
from typing import Dict, List, Any

sys.path.insert(0, "/trading-scanner")

try:
    from backtest_engine import download_and_cache, run_backtest as _run_bt
    _BT_OK = True
except Exception as e:
    print(f"[backtest] WARNING: cannot import backtest_engine — {e}")
    _BT_OK = False

PRICES_DB = "/trading-scanner/data/prices.db"

async def run_backtest(
    strategy: str,
    tickers: List[str],
    start_date: str,
    end_date: str,
    capital: float,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    if not _BT_OK:
        return {"error": "backtest_engine not available", "success": False}

    loop = asyncio.get_event_loop()

    try:
        # Step 1: Download & cache price data
        all_data = await loop.run_in_executor(
            None,
            partial(download_and_cache, tickers, start_date, end_date, PRICES_DB),
        )
        # Step 2: Run backtest
        result = await loop.run_in_executor(
            None,
            partial(
                _run_bt,
                all_data,
                start_date,
                end_date,
                label=strategy,
                initial_capital=capital,
                commission_buy=params.get("commission_buy", 0.0015),
                commission_sell=params.get("commission_sell", 0.0025),
                slippage=params.get("slippage_pct", 0.003),
            ),
        )
        # result is a dict from backtest_engine
        return result if isinstance(result, dict) else {"raw": str(result)}
    except Exception as e:
        return {"error": str(e), "success": False}
