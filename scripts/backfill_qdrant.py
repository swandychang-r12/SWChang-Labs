#!/usr/bin/env python3
"""
SCRIPT-BACKFILL-QDRANT-20260628
Sprint 5F — Task 1: Backfill Qdrant swandy_setups collection
Run: python backfill_qdrant.py [--tickers BBCA BBRI ...] [--days 120]

Uses qdrant_service.py dari deployed app untuk build vector (83-dim cosine).
"""
import sys
import os
import argparse
import logging
from datetime import datetime, timedelta

import numpy as np
import yfinance as yf
import pandas as pd

# Import dari deployed app (jalan di server)
sys.path.insert(0, '/home/aiops/swandy-fund')
try:
    from app.services.qdrant_service import QdrantService
    QDRANT_FROM_SERVICE = True
except ImportError:
    QDRANT_FROM_SERVICE = False

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, UpdateStatus

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger('backfill_qdrant')

# ── Config ────────────────────────────────────────────────────────────────────
QDRANT_HOST   = os.getenv('QDRANT_HOST', 'localhost')
QDRANT_PORT   = int(os.getenv('QDRANT_PORT', 6333))
COLLECTION    = 'swandy_setups'
VECTOR_DIM    = 83
DISTANCE      = Distance.COSINE

IDX_LIQUID_UNIVERSE = [
    'BBCA', 'BBRI', 'BMRI', 'ASII', 'TLKM',
    'ANTM', 'ADRO', 'GOTO', 'BREN', 'AMMN',
    'UNVR', 'ICBP', 'KLBF', 'INDF', 'PGAS',
]

IHSG_TICKER = '^JKSE'
SUFFIX      = '.JK'


# ── Feature Engineering ───────────────────────────────────────────────────────

def normalize_df(df):
    """Flatten MultiIndex columns (yfinance single-ticker quirk) and squeeze to Series-safe."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    # Drop duplicate columns (keep first)
    df = df.loc[:, ~df.columns.duplicated()]
    return df

def compute_features(df: pd.DataFrame, ihsg: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 83-dim features for Qdrant vector.
    Groups: A(13) + B(13) + C(20) + D(13) + E(8) + F(7) + G(5) + extra(4) = 83
    """
    # Normalize MultiIndex from yfinance
    df   = normalize_df(df.copy())
    ihsg = normalize_df(ihsg.copy())

    c = df['Close'].squeeze()
    h = df['High'].squeeze()
    l = df['Low'].squeeze()
    o = df['Open'].squeeze()
    v = df['Volume'].squeeze()


    feat = pd.DataFrame(index=df.index)

    # ── GROUP A: Relative Strength & Momentum (13) ──────────────────────────
    ihsg_c = ihsg['Close'].reindex(df.index, method='ffill')
    feat['rs_vs_ihsg_1d']  = c.pct_change(1)  - ihsg_c.pct_change(1)
    feat['rs_vs_ihsg_5d']  = c.pct_change(5)  - ihsg_c.pct_change(5)
    feat['rs_vs_ihsg_10d'] = c.pct_change(10) - ihsg_c.pct_change(10)
    feat['rs_vs_ihsg_20d'] = c.pct_change(20) - ihsg_c.pct_change(20)
    feat['rs_rank_universe'] = feat['rs_vs_ihsg_5d'].rank(pct=True) * 100
    feat['price_return_1d']  = c.pct_change(1)
    feat['price_return_3d']  = c.pct_change(3)
    feat['price_return_5d']  = c.pct_change(5)
    feat['price_return_8d']  = c.pct_change(8)
    feat['price_return_20d'] = c.pct_change(20)
    feat['roc_5']  = (c / c.shift(5) - 1) * 100
    feat['roc_10'] = (c / c.shift(10) - 1) * 100
    feat['momentum_accel'] = feat['roc_5'] - feat['roc_5'].shift(5)

    # ── GROUP B: Volume & Liquidity (13) ────────────────────────────────────
    vol_ma5  = v.rolling(5).mean()
    vol_ma10 = v.rolling(10).mean()
    vol_ma20 = v.rolling(20).mean()
    feat['vol_ratio_5d']  = v / vol_ma5.replace(0, np.nan)
    feat['vol_ratio_10d'] = v / vol_ma10.replace(0, np.nan)
    feat['vol_ratio_20d'] = v / vol_ma20.replace(0, np.nan)
    feat['idr_volume']     = v * c
    feat['idr_volume_log'] = np.log10(feat['idr_volume'].clip(lower=1))
    feat['vol_trend_5d']   = v.rolling(5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else np.nan, raw=True)
    feat['vol_up_days_ratio'] = v.rolling(5).apply(
        lambda x: (x > vol_ma20.shift(1).reindex(df.index).iloc[-1]).sum() / 5 if len(x) == 5 else np.nan, raw=True)
    obv = (np.sign(c.diff()) * v).cumsum()
    feat['obv_slope_5d']   = obv.diff(5) / 5
    feat['obv_divergence'] = feat['price_return_5d'] - np.sign(feat['obv_slope_5d'])
    vwap = (c * v).rolling(20).sum() / v.rolling(20).sum()
    feat['vwap_deviation'] = (c - vwap) / vwap * 100
    feat['price_vol_corr_5d'] = pd.Series(
        [c.iloc[max(0,i-5):i].pct_change().corr(v.iloc[max(0,i-5):i].pct_change())
         for i in range(len(c))], index=df.index)
    feat['vol_anomaly_flag'] = (feat['vol_ratio_20d'] >= 3.0).astype(float)
    feat['avg_vol_idr_20d'] = feat['idr_volume'].rolling(20).mean()

    # ── GROUP C: Oscillators & Trend (20) ───────────────────────────────────
    # RSI
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    feat['rsi_14'] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
    gain7 = delta.clip(lower=0).rolling(7).mean()
    loss7 = (-delta.clip(upper=0)).rolling(7).mean()
    feat['rsi_7']  = 100 - (100 / (1 + gain7 / loss7.replace(0, np.nan)))
    feat['rsi_zone'] = pd.cut(feat['rsi_14'], bins=[0,40,65,100], labels=[0,1,2]).astype(float)
    # MACD
    ema12 = c.ewm(span=12).mean()
    ema26 = c.ewm(span=26).mean()
    macd_line = ema12 - ema26
    macd_sig  = macd_line.ewm(span=9).mean()
    feat['macd_line']      = macd_line
    feat['macd_signal']    = macd_sig
    feat['macd_histogram'] = macd_line - macd_sig
    cross = (macd_line > macd_sig).astype(int).diff()
    feat['macd_crossover'] = cross.map({1: 1, -1: -1}).fillna(0)
    feat['macd_hist_slope'] = feat['macd_histogram'].diff(3)
    # Stochastic
    low14  = l.rolling(14).min()
    high14 = h.rolling(14).max()
    stoch_k = (c - low14) / (high14 - low14 + 1e-9) * 100
    stoch_d = stoch_k.rolling(3).mean()
    feat['stoch_k'] = stoch_k
    feat['stoch_d'] = stoch_d
    k_cross = (stoch_k > stoch_d).astype(int).diff()
    feat['stoch_crossover'] = k_cross.map({1: 1, -1: -1}).fillna(0)
    # EMA positions
    for p, name in [(5,'ema5'), (10,'ema10'), (20,'ema20'), (50,'ema50'), (200,'ema200')]:
        ema = c.ewm(span=p).mean()
        feat[f'{name}_position'] = (c - ema) / ema * 100
    ema5  = c.ewm(span=5).mean()
    ema10 = c.ewm(span=10).mean()
    ema20 = c.ewm(span=20).mean()
    ema50 = c.ewm(span=50).mean()
    feat['ma_stack'] = ((ema5 > ema10) & (ema10 > ema20) & (ema20 > ema50)).astype(float)
    feat['ema20_ema50_cross'] = np.where(ema20 > ema50, 1, -1).astype(float)
    # CCI
    tp = (h + l + c) / 3
    ma_tp = tp.rolling(14).mean()
    md    = tp.rolling(14).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    feat['cci_14'] = (tp - ma_tp) / (0.015 * md.replace(0, np.nan))
    # Williams %R
    feat['williams_r'] = (high14 - c) / (high14 - low14 + 1e-9) * -100

    # ── GROUP D: Volatility & Structure (13) ────────────────────────────────
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    feat['atr_14']    = atr14
    feat['atr_ratio'] = atr14 / c * 100
    feat['atr_trend'] = atr14 / atr14.shift(5) - 1
    bb_mid   = c.rolling(20).mean()
    bb_std   = c.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    feat['bb_upper']       = bb_upper
    feat['bb_lower']       = bb_lower
    feat['bb_pct_b']       = (c - bb_lower) / (bb_upper - bb_lower + 1e-9)
    feat['bb_width']       = (bb_upper - bb_lower) / bb_mid * 100
    bb_width_pct20 = feat['bb_width'].rolling(20).quantile(0.2)
    feat['bb_squeeze_flag'] = (feat['bb_width'] < bb_width_pct20).astype(float)
    high52 = h.rolling(252).max()
    low52  = l.rolling(252).min()
    feat['w52_position']   = (c - low52) / (high52 - low52 + 1e-9) * 100
    daily_ret = c.pct_change()
    feat['hist_vol_5d']    = daily_ret.rolling(5).std()  * np.sqrt(252)
    feat['hist_vol_20d']   = daily_ret.rolling(20).std() * np.sqrt(252)
    feat['vol_ratio_hvol'] = feat['hist_vol_5d'] / feat['hist_vol_20d'].replace(0, np.nan)
    feat['consolidation_score'] = daily_ret.abs().rolling(20).apply(
        lambda x: 20 - int(np.argmax(x >= 0.03)) if (x >= 0.03).any() else 20, raw=True)

    # ── GROUP E: Market Regime / IHSG (8) ───────────────────────────────────
    ihsg_ma20  = ihsg_c.rolling(20).mean()
    ihsg_ma50  = ihsg_c.rolling(50).mean()
    ihsg_ma200 = ihsg_c.rolling(200).mean()
    feat['ihsg_ma20_regime']  = (ihsg_c > ihsg_ma20).astype(float)
    feat['ihsg_ma50_regime']  = (ihsg_c > ihsg_ma50).astype(float)
    feat['ihsg_ma200_regime'] = (ihsg_c > ihsg_ma200).astype(float)
    feat['ihsg_regime_score'] = feat['ihsg_ma20_regime'] + feat['ihsg_ma50_regime'] + feat['ihsg_ma200_regime']
    feat['ihsg_return_1d'] = ihsg_c.pct_change(1)
    feat['ihsg_return_5d'] = ihsg_c.pct_change(5)
    ihsg_vol = ihsg['Volume'].reindex(df.index, method='ffill')
    feat['ihsg_vol_ratio'] = ihsg_vol / ihsg_vol.rolling(20).mean().replace(0, np.nan)
    cov = daily_ret.rolling(20).cov(ihsg_c.pct_change())
    var = ihsg_c.pct_change().rolling(20).var()
    feat['stock_beta_ihsg'] = cov / var.replace(0, np.nan)

    # ── GROUP F: IDX-Specific (7) ────────────────────────────────────────────
    bar_range = (h - l).replace(0, np.nan)
    feat['foreign_flow_proxy']     = (c - o) / bar_range
    feat['foreign_accumulation_3d'] = feat['foreign_flow_proxy'].rolling(3).sum()
    feat['corp_action_proximity']  = 0.0   # 0 = no known corp action
    feat['corp_action_flag']       = 0.0
    feat['sector_rs_vs_ihsg_5d']   = feat['rs_vs_ihsg_5d']  # proxy = stock RS
    feat['sector_rank']            = 5.0   # default mid-rank
    feat['stock_vs_sector_5d']     = 0.0   # proxy = 0 (no sector index)

    # ── GROUP G: Composite / Engineered (5) ─────────────────────────────────
    feat['vol_rs_combo'] = np.where(
        (feat['vol_ratio_20d'] > 1) & (feat['rs_vs_ihsg_5d'] > 0),
        feat['vol_ratio_20d'] * feat['rs_vs_ihsg_5d'], 0)
    feat['rsi_vol_signal'] = (
        (feat['rsi_14'] < 45) & (feat['vol_ratio_20d'] > 2.0)).astype(float)
    feat['ma_stack_with_volume'] = feat['ma_stack'] * feat['vol_ratio_20d']
    feat['ihsg_filtered_score'] = feat['ihsg_regime_score'] * (
        feat['vol_rs_combo'] + feat['rsi_vol_signal'] + feat['ma_stack_with_volume'])
    feat['setup_quality_score'] = (
        0.35 * feat['rs_rank_universe'].clip(0, 100) / 100 +
        0.30 * feat['vol_ratio_20d'].clip(0, 5) / 5 +
        0.25 * (100 - feat['rsi_14'].clip(0, 100)) / 100 +
        0.10 * feat['bb_pct_b'].clip(0, 1)
    ) * 100

    # ── EXTRA (4): raw price context ─────────────────────────────────────────
    feat['close_norm']  = c / c.rolling(20).mean()
    feat['high_norm']   = h / c.rolling(20).mean()
    feat['low_norm']    = l / c.rolling(20).mean()
    feat['volume_norm'] = np.log1p(v / vol_ma20.replace(0, 1))

    return feat


def build_vector(row: pd.Series) -> list:
    """Convert feature row to 83-dim float list. Clip & fill NaN → 0."""
    vals = row.values.astype(float)
    vals = np.nan_to_num(vals, nan=0.0, posinf=5.0, neginf=-5.0)
    vals = np.clip(vals, -10.0, 10.0)
    assert len(vals) == VECTOR_DIM, f"Expected {VECTOR_DIM} dims, got {len(vals)}"
    return vals.tolist()


# ── Qdrant helpers ────────────────────────────────────────────────────────────

def ensure_collection(client: QdrantClient):
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION not in existing:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=DISTANCE)
        )
        log.info(f"Created collection '{COLLECTION}' (dim={VECTOR_DIM}, {DISTANCE})")
    else:
        log.info(f"Collection '{COLLECTION}' already exists")


def upsert_setups(client: QdrantClient, ticker: str, feat: pd.DataFrame,
                  label_df: pd.DataFrame | None = None):
    """Upsert feature rows as Qdrant points."""
    points = []
    for idx, row in feat.iterrows():
        try:
            vec = build_vector(row)
        except AssertionError as e:
            log.warning(f"{ticker} {idx}: {e} — skip")
            continue

        date_str = idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx)
        # Unique point ID: hash(ticker + date)
        point_id = abs(hash(f"{ticker}_{date_str}")) % (2**31)

        payload = {
            'ticker':               ticker,
            'date':                 date_str,
            'setup_quality_score':  float(row.get('setup_quality_score', 0)),
            'rsi_14':               float(row.get('rsi_14', 0)),
            'vol_ratio_20d':        float(row.get('vol_ratio_20d', 0)),
            'rs_vs_ihsg_5d':        float(row.get('rs_vs_ihsg_5d', 0)),
            'ihsg_regime_score':    float(row.get('ihsg_regime_score', 0)),
            'ma_stack':             bool(row.get('ma_stack', 0)),
            'label_win':            None,   # filled by outcome tracker
            'source':               'backfill_sprint5F',
        }

        # Add outcome label if available (from label_df)
        if label_df is not None and date_str in label_df.index.strftime('%Y-%m-%d').tolist():
            payload['label_win'] = bool(label_df.loc[date_str, 'label_win'])

        points.append(PointStruct(id=point_id, vector=vec, payload=payload))

    if not points:
        log.warning(f"{ticker}: no valid points to upsert")
        return 0

    # Upsert in batches of 100
    total = 0
    for i in range(0, len(points), 100):
        batch = points[i:i+100]
        result = client.upsert(collection_name=COLLECTION, points=batch, wait=True)
        if result.status == UpdateStatus.COMPLETED:
            total += len(batch)
    log.info(f"  {ticker}: upserted {total} points")
    return total


def compute_labels(close: pd.Series, entry_factor: float = 1.0025) -> pd.DataFrame:
    """Label T+3: win if close[+3] >= entry_price * 1.02"""
    entry = close * entry_factor
    future3 = close.shift(-3)
    label_win = (future3 >= entry * 1.02).astype(float)
    return pd.DataFrame({'label_win': label_win}, index=close.index)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Backfill Qdrant swandy_setups')
    parser.add_argument('--tickers', nargs='+', default=IDX_LIQUID_UNIVERSE[:10],
                        help='IDX tickers (without .JK suffix)')
    parser.add_argument('--days', type=int, default=120,
                        help='Lookback days for historical data (default: 120)')
    parser.add_argument('--qdrant-host', default=QDRANT_HOST)
    parser.add_argument('--qdrant-port', type=int, default=QDRANT_PORT)
    parser.add_argument('--dry-run', action='store_true', help='Compute only, no upsert')
    args = parser.parse_args()

    log.info(f"=== Backfill Qdrant | collection={COLLECTION} | dim={VECTOR_DIM} ===")
    log.info(f"Tickers: {args.tickers}")
    log.info(f"Lookback: {args.days} days | Host: {args.qdrant_host}:{args.qdrant_port}")

    # Connect Qdrant
    client = QdrantClient(host=args.qdrant_host, port=args.qdrant_port, timeout=30)
    ensure_collection(client)

    # Download IHSG once
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days + 300)  # extra for MAs
    log.info(f"Downloading IHSG from {start_date.date()} to {end_date.date()}")
    ihsg_df = yf.download(IHSG_TICKER, start=start_date, end=end_date,
                          interval='1d', progress=False, auto_adjust=True)
    ihsg_df = normalize_df(ihsg_df)
    if ihsg_df.empty:
        log.error("Failed to download IHSG data — abort")
        sys.exit(1)

    total_upserted = 0
    failed = []

    for ticker in args.tickers:
        log.info(f"\n── Processing {ticker} ──")
        try:
            yf_ticker = ticker + SUFFIX
            df = yf.download(yf_ticker, start=start_date, end=end_date,
                             interval='1d', progress=False, auto_adjust=True)
            # Normalize MultiIndex
            df = normalize_df(df)
            if df.empty or len(df) < 50:
                log.warning(f"{ticker}: insufficient data ({len(df)} rows) — skip")
                failed.append(ticker)
                continue

            # Compute features
            feat = compute_features(df, ihsg_df)

            # Only keep rows in the lookback window
            cutoff = datetime.now() - timedelta(days=args.days)
            feat = feat[feat.index >= cutoff]
            feat = feat.dropna(how='all')

            # Compute labels (best effort — last 3 rows have NaN)
            label_df = compute_labels(df['Close'])

            log.info(f"  Features computed: {len(feat)} rows, {len(feat.columns)} cols")
            assert feat.shape[1] == VECTOR_DIM, \
                f"Expected {VECTOR_DIM} feature cols, got {feat.shape[1]}"

            if args.dry_run:
                log.info(f"  DRY RUN — skipping upsert")
                log.info(f"  Sample vector (first row): {feat.iloc[0].values[:5]}...")
                continue

            n = upsert_setups(client, ticker, feat, label_df)
            total_upserted += n

        except Exception as e:
            log.error(f"{ticker}: FAILED — {e}", exc_info=True)
            failed.append(ticker)

    # ── Final report ─────────────────────────────────────────────────────────
    log.info(f"\n{'='*50}")
    log.info(f"DONE | Total upserted: {total_upserted} points")
    if failed:
        log.warning(f"Failed tickers: {failed}")

    # Verify collection
    info = client.get_collection(COLLECTION)
    log.info(f"Collection '{COLLECTION}': {info.points_count} total points")
    log.info(f"Vector config: {info.config.params.vectors}")


if __name__ == '__main__':
    main()
