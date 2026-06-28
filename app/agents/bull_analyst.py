from .base_agent import BaseAgent
import json

class BullAnalystAgent(BaseAgent):
    def get_system_prompt(self, ticker: str, market_data: dict) -> str:
        return """Kamu adalah analis saham IDX yang BULLISH dan OPTIMIS. Tugasmu: temukan SEMUA alasan mengapa {ticker} layak DIBELI.
Data tersedia: price action, RSI, MACD, EMA, volume_ratio, broker_flow, ml_probability.
OUTPUT WAJIB JSON SAJA (tanpa markdown):
{{\"stance\": \"BULLISH\", \"confidence\": 0.0-1.0, \"key_points\": [\"poin1\",\"poin2\",\"poin3\"], \"reasoning\": \"200 kata\"}}""".format(ticker=ticker, lookback=60)

    def get_prompt_content(self, ticker: str, market_data: dict) -> str:
        price_data = market_data.get('price_data', {})
        indicators = market_data.get('indicators', {})
        ml_signal = market_data.get('ml_signal', {})
        broker_flow = market_data.get('broker_flow', {})
        
        prompt = """TICKER: {ticker}
DATA SAAT INI:
- Harga terakhir: {close}
- Perubahan hari ini: {change}%
- Volume ratio: {vol_ratio}x (hari ini/avg 20 hari)
- RSI: {rsi}
- MACD histogram: {macd}
- Posisi EMA: {ema20} (20) vs {ema50} (50) - di atas EMA20: {above_ema}
- ML Signal: {ml_action} (probabilitas: {ml_prob}%)
- Flow broker: {broker_signal} (foreign net: {foreign_net} IDR)
- 52w range: {low_52w} - {high_52w}

ANALISIS BULLISH WAJIB:
1. Momentum positif (jika ada)
2. Support kuat dari 52w low
3. Katalis fundamental (jika diketahui)
4. Technical breakout (jika terdeteksi)
5. Volume accumulation
6. Market sentiment IDX bullish
7. Catalyst dari broker flow

Berikan analisis BULLISH yang kuat dengan confidence 0-1.0 dan minimal 3 key points.""".format(
            ticker=ticker,
            close=price_data.get('close_last', 'N/A'),
            change=price_data.get('change_pct', 'N/A'),
            vol_ratio=price_data.get('volume_ratio', 'N/A'),
            rsi=indicators.get('rsi', 'N/A'),
            macd=indicators.get('macd_hist', 'N/A'),
            ema20=indicators.get('ema20', 'N/A'),
            ema50=indicators.get('ema50', 'N/A'),
            above_ema=indicators.get('above_ema20', 'N/A'),
            ml_action=ml_signal.get('action', 'N/A'),
            ml_prob=round(ml_signal.get('probability', 0.5) * 100, 2),
            broker_signal=broker_flow.get('stockbit_signal', 'N/A'),
            foreign_net=broker_flow.get('foreign_net', 'N/A'),
            low_52w=price_data.get('low_52w', 'N/A'),
            high_52w=price_data.get('high_52w', 'N/A')
        )
        return prompt
