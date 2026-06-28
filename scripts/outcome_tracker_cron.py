#!/usr/bin/env python3
"""
SCRIPT-OUTCOME-TRACKER-20260628
Sprint 5F — Task 2: Outcome tracker cron T+3
Run: python outcome_tracker_cron.py
Cron: 30 16 * * 1-5   (after market close 16:30 WIB / 09:30 UTC weekdays)

Flow:
1. Query verdict_outcomes where outcome_filled = FALSE and verdict_date <= today - 3 trading days
2. Fetch close price T+3 via yfinance
3. Calculate outcome: win/loss, return_pct, SL hit
4. Update verdict_outcomes row
5. Also update Qdrant point payload (label_win) for that setup
"""
import sys
import os
import logging
from datetime import datetime, date, timedelta

import psycopg2
import psycopg2.extras
import yfinance as yf
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/aiops/swandy-fund')

from qdrant_client import QdrantClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger('outcome_tracker')

# ── Config ────────────────────────────────────────────────────────────────────
DB_URL       = os.getenv('DATABASE_URL',
    'postgresql://fund_ai_user:fund_ai_pw@10.19.9.240:5432/fund_ai')
QDRANT_HOST  = os.getenv('QDRANT_HOST', 'localhost')
QDRANT_PORT  = int(os.getenv('QDRANT_PORT', 6333))
COLLECTION   = 'swandy_setups'
SUFFIX       = '.JK'

# Win condition: close[T+3] >= entry * 1.02
WIN_TARGET_PCT  = 0.02
ENTRY_SLIPPAGE  = 0.005   # 0.5% slippage on next open
SL_ATR_MULT     = 1.5     # stop_loss = entry - 1.5 * ATR14


# ── Trading calendar helpers ──────────────────────────────────────────────────

def get_trading_days_ago(n: int, ref_date: date = None) -> date:
    """Return date that is n IDX trading days before ref_date (skip weekends)."""
    if ref_date is None:
        ref_date = date.today()
    d = ref_date
    count = 0
    while count < n:
        d -= timedelta(days=1)
        if d.weekday() < 5:   # Mon-Fri only (no IDX holiday calendar for now)
            count += 1
    return d


def trading_days_between(d1: date, d2: date) -> list:
    """List of weekday dates from d1 to d2 inclusive."""
    result = []
    current = d1
    while current <= d2:
        if current.weekday() < 5:
            result.append(current)
        current += timedelta(days=1)
    return result


# ── Price fetcher ─────────────────────────────────────────────────────────────

def fetch_price_range(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Fetch OHLCV for ticker (adds .JK) from start to end+1 day."""
    yf_ticker = ticker + SUFFIX
    df = yf.download(
        yf_ticker,
        start=start.strftime('%Y-%m-%d'),
        end=(end + timedelta(days=5)).strftime('%Y-%m-%d'),
        interval='1d', progress=False, auto_adjust=True
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).date
    return df


def compute_outcome(entry_price: float, atr14: float,
                    df: pd.DataFrame, entry_date: date) -> dict:
    """
    Compute T+3 outcome.
    - entry_price: actual entry (next open + slippage, supplied from DB)
    - atr14: ATR at signal date
    - df: OHLCV from signal date onward
    - entry_date: date of actual entry
    Returns dict with outcome fields.
    """
    stop_loss = entry_price * (1 - 0) - SL_ATR_MULT * atr14  # entry - 1.5*ATR
    # Find T+1, T+2, T+3 after entry
    future_dates = [d for d in df.index if d > entry_date][:3]

    if len(future_dates) < 1:
        return {'outcome_filled': False, 'error': 'no future prices'}

    close_t3 = df.loc[future_dates[-1], 'Close'] if len(future_dates) >= 3 else None
    t3_date  = future_dates[-1]

    # Check SL hit (any intraday low in T+1..T+3)
    sl_hit = False
    for d in future_dates:
        if df.loc[d, 'Low'] <= stop_loss:
            sl_hit = True
            break

    # Win condition
    win = False
    if close_t3 is not None and not sl_hit:
        win = (close_t3 >= entry_price * (1 + WIN_TARGET_PCT))

    # Return pct at T+3 (or at SL hit day)
    exit_price = close_t3 if close_t3 is not None else entry_price
    return_pct = (exit_price - entry_price) / entry_price * 100

    return {
        'outcome_filled':    True,
        'outcome_date':      t3_date,
        'outcome_price':     float(exit_price) if exit_price else None,
        'outcome_return_pct': float(return_pct),
        'label_win':         bool(win),
        'sl_hit':            sl_hit,
        'stop_loss_price':   float(stop_loss),
        'filled_at':         datetime.utcnow(),
        'error':             None,
    }


# ── DB helpers ────────────────────────────────────────────────────────────────

def get_pending_verdicts(conn) -> list:
    """
    Fetch rows from verdict_outcomes that need outcome filling.
    Condition: outcome_filled = FALSE AND verdict_date <= T-3 (trading days).
    """
    cutoff = get_trading_days_ago(3)
    query = """
        SELECT
            id,
            ticker,
            verdict_date,
            entry_price,
            atr_14,
            stop_loss_price,
            signal_id
        FROM verdict_outcomes
        WHERE outcome_filled = FALSE
          AND verdict_date <= %s
        ORDER BY verdict_date ASC
        LIMIT 50
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(query, (cutoff,))
        return cur.fetchall()


def update_verdict_outcome(conn, row_id: int, outcome: dict):
    """Update verdict_outcomes row with computed outcome."""
    query = """
        UPDATE verdict_outcomes SET
            outcome_date      = %s,
            outcome_price     = %s,
            outcome_return_pct = %s,
            label_win         = %s,
            sl_hit            = %s,
            outcome_filled    = TRUE,
            filled_at         = %s
        WHERE id = %s
    """
    with conn.cursor() as cur:
        cur.execute(query, (
            outcome.get('outcome_date'),
            outcome.get('outcome_price'),
            outcome.get('outcome_return_pct'),
            outcome.get('label_win'),
            outcome.get('sl_hit'),
            outcome.get('filled_at'),
            row_id,
        ))
    conn.commit()


# ── Qdrant update ─────────────────────────────────────────────────────────────

def update_qdrant_label(qdrant: QdrantClient, ticker: str,
                        verdict_date: date, label_win: bool):
    """Update label_win in Qdrant payload for this setup point."""
    date_str = verdict_date.strftime('%Y-%m-%d')
    point_id = abs(hash(f"{ticker}_{date_str}")) % (2**31)
    try:
        qdrant.set_payload(
            collection_name=COLLECTION,
            payload={'label_win': label_win, 'outcome_filled': True},
            points=[point_id],
            wait=False,
        )
        log.debug(f"  Qdrant updated: {ticker} {date_str} → win={label_win}")
    except Exception as e:
        log.warning(f"  Qdrant update failed for {ticker} {date_str}: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("=== Outcome Tracker T+3 | swandy-fund ===")
    log.info(f"Reference date: {date.today()} | cutoff: {get_trading_days_ago(3)}")

    # Connect
    try:
        conn   = psycopg2.connect(DB_URL)
        qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=20)
    except Exception as e:
        log.error(f"Connection failed: {e}")
        sys.exit(1)

    # Fetch pending
    pending = get_pending_verdicts(conn)
    log.info(f"Pending verdicts to fill: {len(pending)}")

    if not pending:
        log.info("Nothing to do — all outcomes up to date")
        conn.close()
        return

    filled  = 0
    skipped = 0
    errors  = []

    # Cache price data per ticker (avoid duplicate downloads)
    price_cache: dict[str, pd.DataFrame] = {}

    for row in pending:
        ticker       = row['ticker']
        verdict_date = row['verdict_date']   # date object
        entry_price  = float(row['entry_price'])
        atr14        = float(row['atr_14']) if row['atr_14'] else 0.0

        log.info(f"  {ticker} | verdict={verdict_date} | entry={entry_price:.0f}")

        try:
            # Download price if not cached
            if ticker not in price_cache:
                start = verdict_date - timedelta(days=2)
                end   = date.today()
                price_cache[ticker] = fetch_price_range(ticker, start, end)

            df = price_cache[ticker]
            if df.empty:
                log.warning(f"  {ticker}: no price data — skip")
                skipped += 1
                continue

            # Compute outcome
            outcome = compute_outcome(entry_price, atr14, df, verdict_date)
            if not outcome['outcome_filled']:
                log.warning(f"  {ticker} {verdict_date}: {outcome.get('error')} — skip")
                skipped += 1
                continue

            # Update DB
            update_verdict_outcome(conn, row['id'], outcome)
            filled += 1

            # Update Qdrant label
            update_qdrant_label(qdrant, ticker, verdict_date, outcome['label_win'])

            log.info(f"  → outcome: win={outcome['label_win']} | "
                     f"ret={outcome['outcome_return_pct']:.2f}% | "
                     f"sl_hit={outcome['sl_hit']}")

        except Exception as e:
            log.error(f"  {ticker} {verdict_date}: ERROR — {e}", exc_info=True)
            errors.append(f"{ticker}_{verdict_date}")

    # ── Summary ──────────────────────────────────────────────────────────────
    log.info(f"\n{'='*50}")
    log.info(f"DONE | filled={filled} | skipped={skipped} | errors={len(errors)}")
    if errors:
        log.warning(f"Errored: {errors}")

    conn.close()
    log.info("DB connection closed")


if __name__ == '__main__':
    main()
