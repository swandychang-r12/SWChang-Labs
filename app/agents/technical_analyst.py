from .base_agent import BaseAgent
import json

class TechnicalAnalystAgent(BaseAgent):
    def get_system_prompt(self, ticker: str, market_data: dict) -> str:
        return """Kamu adalah analis saham IDX yang fokus MURNI pada chart dan indikator teknikal. Tugasmu adalah menganalisis {ticker} berdasarkan data teknikal saja.
OUTPUT WAJIB JSON SAJA (tanpa markdown):
{{\"stance\": \"BULLISH\"|\"BEARISH\"|\"NEUTRAL\", \"confidence\": \"VERY_HIGH\"|..., \"key_points\": [\"poin1\",\"poin2\",\"poin3\"], \"reasoning\": \"200 kata\"}}""".format(ticker=ticker, lookback=60)

    def get_prompt_content(self, ticker: str, market_data: dict) -> str:
        price_data = market_data.get('price_data', {})
        indicators = market_data.get('indicators', {})
        
        prompt = """TICKER: {ticker}
DATA TEKNIKAL SAAT INI:
- Harga terakhir: {close}
- Perubahan hari ini: {change}%
- Volume ratio: {vol_ratio}x (hari ini/avg 20 hari)
- RSI: {rsi}
- MACD histogram: {macd}
- Posisi EMA: {ema20} (20) vs {ema50} (50) - di atas EMA20: {above_ema}
- Bollinger Band position (0=lower, 1=upper): {bb_pos}
- Average True Range (ATR): {atr}
- ADX: {adx}
- 52w range: {low_52w} - {high_52w}

ANALISIS TEKNIKAL WAJIB:
1. Trend jangka pendek/menengah (berdasarkan EMA/ADX)
2. Potensi pembalikan/penerusan (berdasarkan RSI/MACD/Candle Pattern jika ada)
3. Level support dan resistance (dari 52w range atau price action)
4. Volatilitas (dari ATR/BB)
5. Volume profile

Berikan analisis teknikal yang NETRAL atau dengan stance (BULLISH/BEARISH) yang jelas, confidence 0-1.0, dan minimal 3 key points.""".format(
            ticker=ticker,
            close=price_data.get('close_last', 'N/A'),
            change=price_data.get('change_pct', 'N/A'),
            vol_ratio=price_data.get('volume_ratio', 'N/A'),
            rsi=indicators.get('rsi', 'N/A'),
            macd=indicators.get('macd_hist', 'N/A'),
            ema20=indicators.get('ema20', 'N/A'),
            ema50=indicators.get('ema50', 'N/A'),
            above_ema=indicators.get('above_ema20', 'N/A'),
            bb_pos=indicators.get('bb_position', 'N/A'),
            atr=indicators.get('atr', 'N/A'),
            adx=indicators.get('adx', 'N/A'),
            low_52w=price_data.get('low_52w', 'N/A'),
            high_52w=price_data.get('high_52w', 'N/A')
        )
        return prompt
