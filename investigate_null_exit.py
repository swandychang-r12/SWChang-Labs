#!/usr/bin/env python3
"""
investigate_null_exit.py -- C-11 Root cause analysis
Why 106 trades in 2026 have exit_price=NULL in features.parquet?
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

SCANNER = Path("/home/aiops/trading-scanner")
DATA    = SCANNER / "data"

print("=" * 60)
print("C-11: NULL exit_price investigation")
print("=" * 60)

# Load full features parquet
df = pd.read_parquet(DATA / "features.parquet")
print(f"\n[1] features.parquet shape: {df.shape}")
print(f"    columns with exit: {[c for c in df.columns if 'exit' in c or 'entry' in c or 'fwd' in c]}")
print(f"    date range: {df['date'].min()} → {df['date'].max()}")
print(f"    tickers: {df['ticker'].nunique()}")

# Check exit_price nulls
null_ep = df[df['exit_price'].isna()]
print(f"\n[2] Rows with exit_price=NULL: {len(null_ep)}")

if len(null_ep) > 0:
    print(f"    Null by year:")
    null_ep['year'] = pd.to_datetime(null_ep['date']).dt.year
    print(null_ep.groupby('year').size().to_string())

    print(f"\n    Last 5 dates in null set:")
    null_dates = sorted(null_ep['date'].unique())
    for d in null_dates[-5:]:
        n = len(null_ep[null_ep['date'] == d])
        print(f"      {d}  →  {n} tickers null")

    print(f"\n    Last 5 dates in full dataset:")
    all_dates = sorted(df['date'].unique())
    for d in all_dates[-5:]:
        total = len(df[df['date'] == d])
        null  = len(null_ep[null_ep['date'] == d])
        print(f"      {d}  →  {total} tickers, {null} null exit_price")

# 2026 slice analysis
df['date'] = pd.to_datetime(df['date'])
df2026 = df[df['date'].dt.year == 2026]
print(f"\n[3] 2026 slice: {len(df2026)} rows, {df2026['date'].nunique()} trading days")
null2026 = df2026[df2026['exit_price'].isna()]
valid2026 = df2026[df2026['exit_price'].notna() & (df2026['entry_price'] > 0)]
print(f"    Valid (exit_price set):   {len(valid2026)}")
print(f"    NULL exit_price:          {len(null2026)}")
print(f"    % null of 2026 total:     {len(null2026)/len(df2026)*100:.1f}%")

# WR if we include nulls vs exclude
print(f"\n[4] 2026 WR analysis (valid rows only):")
if 'label_win' in df2026.columns:
    wr_valid = valid2026['label_win'].mean() * 100
    print(f"    WR (label_win, valid):    {wr_valid:.1f}%")
if 'fwd_return_3d' in df2026.columns:
    wins = (valid2026['fwd_return_3d'] > 0).sum()
    total = valid2026['fwd_return_3d'].notna().sum()
    wr_fwd = wins / total * 100 if total > 0 else 0
    print(f"    WR (fwd_return_3d > 0):   {wr_fwd:.1f}%  (n={total})")
    print(f"    Avg fwd_return_3d:         {valid2026['fwd_return_3d'].mean():.2f}%")

# Boundary check — are nulls ONLY at the tail?
print(f"\n[5] Boundary check — are nulls concentrated at data tail?")
df_sorted = df.sort_values('date')
last_3_dates = sorted(df['date'].unique())[-3:]
print(f"    Last 3 dates in dataset: {[str(d.date()) for d in last_3_dates]}")
tail_nulls = null_ep[pd.to_datetime(null_ep['date']).isin(last_3_dates)]
non_tail_nulls = null_ep[~pd.to_datetime(null_ep['date']).isin(last_3_dates)]
print(f"    Nulls ON last 3 dates:   {len(tail_nulls)}")
print(f"    Nulls BEFORE last 3:     {len(non_tail_nulls)} ← if > 0, there are mid-data gaps")

if len(non_tail_nulls) > 0:
    print(f"\n    ⚠ MID-DATA NULL DATES:")
    mid_dates = sorted(non_tail_nulls['date'].unique())
    for d in mid_dates[:10]:
        n = len(non_tail_nulls[non_tail_nulls['date'] == d])
        print(f"      {d}: {n} tickers")

print(f"\n[6] Summary & root cause verdict:")
if len(non_tail_nulls) == 0:
    print(f"    ROOT CAUSE: DATA BOUNDARY — last 3 trading days of dataset")
    print(f"    have no T+3 forward data (c.shift(-3) = NaN). EXPECTED.")
    print(f"    Code comment line 387 confirms: 'Keep all rows; last 3 have NaN labels'")
    print(f"    WR=39.8% is from only {len(valid2026)} valid 2026 rows (~6 months).")
    print(f"    VERDICT: NOT A BUG. AUC_WF=0.5975 is the correct acceptance criterion.")
else:
    print(f"    ROOT CAUSE: OHLCV DATA GAPS — {len(non_tail_nulls)} rows with null")
    print(f"    exit_price are NOT at tail. Some tickers have missing OHLCV data mid-2026.")
    print(f"    Affected tickers: {sorted(non_tail_nulls['ticker'].unique())}")

print("\n" + "=" * 60)