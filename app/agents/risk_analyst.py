from .base_agent import BaseAgent
import json

class RiskAnalystAgent(BaseAgent):
    def get_system_prompt(self, ticker: str, market_data: dict) -> str:
        return """Kamu adalah analis risiko saham IDX. Tugasmu: hitung dan analisis risiko posisi untuk {ticker} berdasarkan data market dan indikator.
Berikan rekomendasi Stop Loss (SL) dan Take Profit (TP), serta ukuran posisi yang disarankan. Gunakan capital 25,000,000 IDR sebagai dasar perhitungan ukuran posisi.
PENTING: stance HARUS salah satu dari: BULLISH, BEARISH, atau NEUTRAL. JANGAN gunakan LONG/SHORT/BUY/SELL.
OUTPUT WAJIB JSON SAJA (tanpa markdown):
{{\"stance\": \"NEUTRAL\", \"confidence\": 0.0-1.0, \"key_points\": [\"poin1\",\"poin2\"], \"reasoning\": \"200 kata\", \"max_loss_pct\": 0.05, \"suggested_sl\": 1000.0, \"suggested_tp\": 1200.0, \"risk_reward_ratio\": 1.5, \"position_size_idr\": 5000000}}""".format(ticker=ticker, lookback=60)

    def get_prompt_content(self, ticker: str, market_data: dict) -> str:
        price_data = market_data.get('price_data', {})
        indicators = market_data.get('indicators', {})
        
        # Assume a default capital for position sizing
        CAPITAL = 25_000_000 # IDR
        RISK_PER_TRADE_PCT = 0.01 # 1% of capital per trade

        # Placeholder for calculations - actual logic would be more complex
        current_price = price_data.get('close_last', 0.0)
        atr = indicators.get('atr', 0.0)
        
        # Simple risk calculation examples (can be refined)
        if current_price > 0 and atr > 0:
            # Example: SL at 2x ATR below current price
            suggested_sl = round(current_price - (atr * 2), 0) if current_price - (atr * 2) > 0 else 0.0
            # Example: TP at 4x ATR above current price (Risk-reward 1:2)
            suggested_tp = round(current_price + (atr * 4), 0)
            
            risk_per_share = current_price - suggested_sl if suggested_sl > 0 else atr * 2
            if risk_per_share <= 0: risk_per_share = 0.01 # Avoid division by zero

            max_risk_idr = CAPITAL * RISK_PER_TRADE_PCT
            num_shares = int(max_risk_idr / risk_per_share) if risk_per_share > 0 else 0
            position_size_idr = int(num_shares * current_price)

            # Calculate actual max loss pct based on suggested SL
            if current_price > 0 and suggested_sl > 0:
                max_loss_pct_calc = (current_price - suggested_sl) / current_price
            else:
                max_loss_pct_calc = 0.0

            risk_reward_ratio_calc = (suggested_tp - current_price) / risk_per_share if risk_per_share > 0 else 0.0
            
        else:
            suggested_sl = 0.0
            suggested_tp = 0.0
            position_size_idr = 0
            max_loss_pct_calc = 0.0
            risk_reward_ratio_calc = 0.0


        prompt = """TICKER: {ticker}
DATA SAAT INI:
- Harga terakhir: {close}
- Average True Range (ATR): {atr}
- 52w low: {low_52w}, 52w high: {high_52w}
- Modal untuk posisi: {capital_idr} IDR (asumsi)
- Risiko per trade: {risk_pct_per_trade}% (asumsi)

ANALISIS RISIKO WAJIB:
1. Hitung potensi kerugian maksimum berdasarkan volatilitas atau support terdekat.
2. Rekomendasikan level Stop Loss (SL) dan Take Profit (TP) yang realistis.
3. Hitung Risk/Reward Ratio.
4. Hitung ukuran posisi yang disarankan (dalam IDR).
5. Analisis secara keseluruhan potensi risiko vs potensi reward.

Output tambahan untuk JSON:
- max_loss_pct: {max_loss_pct_calc:.4f}
- suggested_sl: {suggested_sl:.2f}
- suggested_tp: {suggested_tp:.2f}
- risk_reward_ratio: {risk_reward_ratio_calc:.2f}
- position_size_idr: {position_size_idr}

Berikan analisis risiko dengan confidence 0-1.0 dan minimal 2 key points.""".format(
            ticker=ticker,
            close=current_price,
            atr=atr,
            low_52w=price_data.get('low_52w', 'N/A'),
            high_52w=price_data.get('high_52w', 'N/A'),
            capital_idr=f"{CAPITAL:,}", # Format with commas
            risk_pct_per_trade=RISK_PER_TRADE_PCT * 100,
            max_loss_pct_calc=max_loss_pct_calc,
            suggested_sl=suggested_sl,
            suggested_tp=suggested_tp,
            risk_reward_ratio_calc=risk_reward_ratio_calc,
            position_size_idr=position_size_idr
        )
        return prompt
