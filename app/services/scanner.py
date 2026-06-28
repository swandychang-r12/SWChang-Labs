import sys, asyncio, traceback
from functools import partial
from typing import List, Dict, Any

sys.path.insert(0, "/trading-scanner")

try:
    from trading_scanner import scan_r12
    _SCANNER_OK = True
except Exception as e:
    print(f"[scanner] WARNING: cannot import scan_r12 — {e}")
    _SCANNER_OK = False

async def run_scanner(
    universe: List[str],
    min_score: float = 0.52,
    min_volume_ratio: float = 1.5,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    if not _SCANNER_OK:
        return []

    loop = asyncio.get_event_loop()
    results: List[Dict] = []

    async def _scan_one(ticker: str):
        try:
            raw = await loop.run_in_executor(None, scan_r12, ticker)
            if raw is None:
                return
            # scan_r12 may return a dict or an object with __dict__
            d = raw if isinstance(raw, dict) else vars(raw)
            ml_prob = float(d.get("ml_prob", d.get("score", 0)))
            vol_ratio = float(d.get("vol_ratio", d.get("volume_ratio", 0)))
            if ml_prob >= min_score and vol_ratio >= min_volume_ratio:
                results.append({
                    "ticker":       ticker,
                    "price":        float(d.get("close", d.get("price", 0))),
                    "change_pct":   float(d.get("chg_pct", d.get("change_pct", 0))),
                    "ml_score":     ml_prob,
                    "volume_ratio": vol_ratio,
                    "action":       d.get("action", d.get("signal", "HOLD")),
                    "rsi":          float(d.get("rsi", 50)),
                    "broker_flow":  d.get("stockbit_signal", d.get("broker_flow", "")),
                    "raw":          {k: v for k, v in d.items()
                                     if k not in ("_sa_instance_state",)},
                })
        except Exception:
            traceback.print_exc()

    # Run all tickers concurrently (thread pool)
    await asyncio.gather(*[_scan_one(t) for t in universe])
    results.sort(key=lambda x: x["ml_score"], reverse=True)
    return results[:limit]
