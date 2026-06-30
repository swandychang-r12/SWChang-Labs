#!/usr/bin/env python3
"""
ml_build_dashboard.py -- ML Paper Trading Dashboard Generator
Phase 4: C-10 | 2026-06-30
Generates /home/aiops/trading-scanner/ml_dashboard.html (served via nginx :8088)
Usage: python3 ml_build_dashboard.py
       or cron: 30 10 * * 1-5 (after market close)
"""
import json
import sys
import urllib.request
from datetime import datetime

API = "http://localhost:8089"
OUT = "/home/aiops/trading-scanner/ml_dashboard.html"

WIB = "Asia/Jakarta"

def fetch(path: str) -> dict:
    try:
        with urllib.request.urlopen(f"{API}{path}", timeout=10) as r:
            return json.loads(r.read())
    except Exception as e:
        print(f"[WARN] fetch {path} failed: {e}")
        return {}


def fmt_pct(v) -> str:
    if v is None: return "-"
    color = "#4ade80" if float(v) >= 0 else "#f87171"
    return f'<span style="color:{color}">{float(v):+.2f}%</span>'


def fmt_price(v) -> str:
    if v is None: return "-"
    return f"{float(v):,.0f}"


def signal_bar(prob: float) -> str:
    pct = int(prob * 100)
    color = "#4ade80" if prob >= 0.65 else "#facc15" if prob >= 0.5 else "#94a3b8"
    return (f'<div style="background:#1e293b;border-radius:4px;height:8px;width:100%">'
            f'<div style="background:{color};width:{pct}%;height:8px;border-radius:4px"></div></div>'
            f'<small style="color:{color}">{prob:.1%}</small>')


def build_html(signals: dict, trades: dict, stats: dict) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M WIB")
    scan_date = signals.get("data", {}).get("scan_date", "-")
    total_buy = signals.get("data", {}).get("total_buy", 0)
    total_scanned = signals.get("data", {}).get("total_scanned", 0)
    sig_rows = signals.get("data", {}).get("signals", [])

    all_trades = trades.get("data", {}).get("trades", [])
    open_trades = [t for t in all_trades if t["status"] == "open"]
    closed_trades = [t for t in all_trades if t["status"] in ("closed", "stopped")]

    st = stats.get("data", {})

    # Signals table
    sig_html = ""
    if not sig_rows:
        sig_html = '<tr><td colspan="4" style="text-align:center;color:#94a3b8">No buy signals today</td></tr>'
    for r in sig_rows:
        sig_html += (
            f'<tr>'
            f'<td><b style="color:#38bdf8">{r["ticker"]}</b></td>'
            f'<td>{signal_bar(r["probability"])}</td>'
            f'<td><span class="badge buy">BUY</span></td>'
            f'<td style="color:#94a3b8">{r.get("as_of","-")}</td>'
            f'</tr>'
        )

    # Open trades table
    open_html = ""
    if not open_trades:
        open_html = '<tr><td colspan="5" style="text-align:center;color:#94a3b8">No open trades</td></tr>'
    for t in open_trades:
        open_html += (
            f'<tr>'
            f'<td><b style="color:#38bdf8">{t["ticker"]}</b></td>'
            f'<td>{t["entry_date"]}</td>'
            f'<td>{fmt_price(t["entry_price"])}</td>'
            f'<td style="color:#f87171">{fmt_price(t["sl_price"])}</td>'
            f'<td style="color:#94a3b8">{t["signal_prob"]:.1%}</td>'
            f'</tr>'
        )

    # Closed trades table
    closed_html = ""
    if not closed_trades:
        closed_html = '<tr><td colspan="6" style="text-align:center;color:#94a3b8">No closed trades yet</td></tr>'
    for t in closed_trades[:30]:
        reason_color = "#f87171" if t.get("exit_reason") == "SL" else "#94a3b8"
        closed_html += (
            f'<tr>'
            f'<td><b style="color:#38bdf8">{t["ticker"]}</b></td>'
            f'<td>{t["entry_date"]} → {t.get("exit_date","-")}</td>'
            f'<td>{fmt_price(t["entry_price"])}</td>'
            f'<td>{fmt_price(t["exit_price"])}</td>'
            f'<td>{fmt_pct(t["return_pct"])}</td>'
            f'<td><span style="color:{reason_color}">{t.get("exit_reason","-")}</span></td>'
            f'</tr>'
        )

    wr_color = "#4ade80" if st.get("win_rate") and st["win_rate"] >= 50 else "#f87171"

    return f"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="300">
<title>ML Paper Trading | xgb_v6 | swandy-fund</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0f172a; color: #e2e8f0; font-family: 'Segoe UI', sans-serif; padding: 20px; }}
  h1 {{ color: #38bdf8; font-size: 1.4rem; margin-bottom: 4px; }}
  .meta {{ color: #64748b; font-size: 0.8rem; margin-bottom: 20px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 24px; }}
  .card {{ background: #1e293b; border-radius: 8px; padding: 16px; }}
  .card .label {{ color: #64748b; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; }}
  .card .value {{ font-size: 1.6rem; font-weight: 700; margin-top: 4px; }}
  .section {{ margin-bottom: 28px; }}
  .section h2 {{ color: #cbd5e1; font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.08em;
                 border-bottom: 1px solid #334155; padding-bottom: 6px; margin-bottom: 12px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.875rem; }}
  th {{ color: #64748b; text-align: left; padding: 6px 10px; font-weight: 500; font-size: 0.75rem;
        text-transform: uppercase; letter-spacing: 0.05em; }}
  td {{ padding: 8px 10px; border-bottom: 1px solid #1e293b; }}
  tr:hover td {{ background: #1e293b; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.72rem;
             font-weight: 600; text-transform: uppercase; }}
  .badge.buy {{ background: #14532d; color: #4ade80; }}
  .model-tag {{ display: inline-block; background: #1e3a5f; color: #38bdf8;
                padding: 3px 10px; border-radius: 4px; font-size: 0.78rem; }}
  footer {{ color: #334155; font-size: 0.75rem; margin-top: 24px; }}
</style>
</head>
<body>

<h1>📈 ML Paper Trading Dashboard</h1>
<div class="meta">
  Last updated: {now} &nbsp;|&nbsp;
  Scan date: {scan_date} &nbsp;|&nbsp;
  <span class="model-tag">xgb_v6 &nbsp;AUC_WF={st.get("auc_wf", 0.5975)}</span>
</div>

<div class="grid">
  <div class="card">
    <div class="label">Buy Signals</div>
    <div class="value" style="color:#4ade80">{total_buy}</div>
    <div style="color:#64748b;font-size:0.78rem">of {total_scanned} scanned</div>
  </div>
  <div class="card">
    <div class="label">Open Trades</div>
    <div class="value" style="color:#38bdf8">{len(open_trades)}</div>
    <div style="color:#64748b;font-size:0.78rem">max 5</div>
  </div>
  <div class="card">
    <div class="label">Total Closed</div>
    <div class="value">{st.get("total_trades", 0)}</div>
  </div>
  <div class="card">
    <div class="label">Win Rate</div>
    <div class="value" style="color:{wr_color}">{f"{st['win_rate']:.1f}%" if st.get("win_rate") is not None else "-"}</div>
  </div>
  <div class="card">
    <div class="label">Avg Return</div>
    <div class="value">{fmt_pct(st.get("avg_return_pct"))}</div>
  </div>
  <div class="card">
    <div class="label">Sharpe</div>
    <div class="value">{f"{st['sharpe']:.2f}" if st.get("sharpe") is not None else "-"}</div>
  </div>
</div>

<div class="section">
  <h2>🔍 Today\'s Signals</h2>
  <table>
    <thead><tr><th>Ticker</th><th>Probability</th><th>Signal</th><th>As Of</th></tr></thead>
    <tbody>{sig_html}</tbody>
  </table>
</div>

<div class="section">
  <h2>📂 Open Paper Trades</h2>
  <table>
    <thead><tr><th>Ticker</th><th>Entry Date</th><th>Entry Price</th><th>Stop Loss</th><th>Prob</th></tr></thead>
    <tbody>{open_html}</tbody>
  </table>
</div>

<div class="section">
  <h2>📊 Closed Trade History</h2>
  <table>
    <thead><tr><th>Ticker</th><th>Entry → Exit</th><th>Entry</th><th>Exit</th><th>Return</th><th>Reason</th></tr></thead>
    <tbody>{closed_html}</tbody>
  </table>
</div>

<footer>
  swandy-fund R12 | Phase 4 | AUC_WF={st.get("auc_wf",0.5975)} |
  SL trades: {st.get("sl_trades",0)} | T3 trades: {st.get("t3_trades",0)} |
  Auto-refresh: 5min
</footer>
</body>
</html>"""


def main():
    print("[ml_dashboard] fetching data from API...")
    signals = fetch("/api/v2/signal/scan")
    trades  = fetch("/api/v2/paper/trades?limit=200")
    stats   = fetch("/api/v2/paper/stats")

    html = build_html(signals, trades, stats)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[ml_dashboard] written to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())