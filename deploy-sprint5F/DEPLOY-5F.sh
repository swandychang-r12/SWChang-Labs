#!/bin/bash
# DEPLOY-SPRINT-5F.sh | 2026-06-28 | swandy-fund
# Run from: /home/aiops/swandy-fund/
# =======================================================
set -e

APP=/home/aiops/swandy-fund
SCRIPTS=/home/aiops/swandy-fund/scripts
echo "=== DEPLOY Sprint 5F ==="

# ── 1. Copy files ───────────────────────────────────────
mkdir -p $SCRIPTS
cp scripts/backfill_qdrant.py      $SCRIPTS/
cp scripts/outcome_tracker_cron.py $SCRIPTS/
cp app/routers/portfolio.py        $APP/app/routers/
cp app/services/similar_context_service.py $APP/app/services/

echo "[OK] Files copied"

# ── 2. Install deps (jika belum) ────────────────────────
pip install qdrant-client psycopg2-binary --quiet
echo "[OK] Deps installed"

# ── 3. Register portfolio router di main.py ─────────────
# Tambahkan 2 baris ini di app/main.py (MANUAL jika belum ada):
# from app.routers import portfolio
# app.include_router(portfolio.router)
echo ""
echo "[!] MANUAL STEP — Tambahkan di app/main.py jika belum ada:"
echo "    from app.routers import portfolio"
echo "    app.include_router(portfolio.router)"
echo ""

# ── 4. Setup cron T+3 ───────────────────────────────────
# Weekdays 16:30 WIB = 09:30 UTC
CRON_JOB="30 9 * * 1-5 cd $APP && python $SCRIPTS/outcome_tracker_cron.py >> /tmp/outcome_tracker.log 2>&1"
( crontab -l 2>/dev/null | grep -v "outcome_tracker_cron"; echo "$CRON_JOB" ) | crontab -
echo "[OK] Cron T+3 registered: '30 9 * * 1-5'"
crontab -l | grep outcome_tracker

# ── 5. Rebuild Docker (kalau main.py diubah) ────────────
echo ""
echo "[!] Jika main.py diubah → rebuild Docker:"
echo "    docker compose build app && docker compose up -d app"

# ── 6. Run backfill ─────────────────────────────────────
echo ""
echo "=== RUNNING BACKFILL ==="
cd $APP
python $SCRIPTS/backfill_qdrant.py --tickers BBCA BBRI BMRI ASII TLKM ANTM ADRO GOTO BREN AMMN --days 90

echo ""
echo "=== Sprint 5F Deploy DONE ==="
