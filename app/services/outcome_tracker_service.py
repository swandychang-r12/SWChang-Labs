"""
Wrapper agar outcome_tracker_cron.py bisa dipanggil oleh APScheduler di main.py.
"""
import logging
log = logging.getLogger(__name__)

async def fill_pending_outcomes():
    """APScheduler job: fill T+3 outcomes (runs at 16:30 WIB weekdays)."""
    try:
        import sys, os
        sys.path.insert(0, "/home/aiops/swandy-fund")
        from scripts.outcome_tracker_cron import main as _main
        import asyncio
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _main)
        log.info("[outcome_tracker] T+3 fill complete")
    except Exception as e:
        log.error(f"[outcome_tracker] failed: {e}", exc_info=True)
