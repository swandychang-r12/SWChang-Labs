from .base_agent import BaseAgent
import json

class OrchestratorAgent(BaseAgent):
    def get_system_prompt(self, ticker: str, market_data: dict) -> str:
        base = """Kamu adalah Chief Investment Officer IDX (Bursa Efek Indonesia). Baca laporan 4 analis + XGBoost ML signal, buat KEPUTUSAN FINAL.
BOBOT: Technical 30% | Bull 25% | Bear 25% | Risk 20%.
ML SIGNAL (XGBoost v5, backtested): WAJIB dijadikan anchor. Jika ML prob >= 0.70 -> bias BUY. Jika ML prob <= 0.30 -> bias SELL. Jika MEDIUM -> andalkan LLM agents.
KONTEKS IDX: ARA/ARB limit 25%/35%, jam trading 09:00-15:00 WIB, T+2 settlement, liquidity IDX bisa tipis.
OUTPUT WAJIB JSON SAJA (tanpa markdown):
{{\"action\": \"STRONG_BUY\"|\"BUY\"|\"HOLD\"|\"SELL\"|\"STRONG_SELL\", \"stance\": \"BULLISH\"|\"BEARISH\"|\"NEUTRAL\", \"confidence\": 0.0-1.0, \"entry_price\": 1000.0, \"stop_loss\": 950.0, \"take_profit\": 1200.0, \"hold_days\": 30, \"ml_anchor\": \"bagaimana ML signal mempengaruhi keputusan ini\", \"executive_summary\": \"150 kata Bahasa Indonesia\"}}"""

        # Inject similar_context dari Qdrant jika tersedia
        similar_ctx = market_data.get("similar_context", "")
        if similar_ctx:
            base += f"\n\n{similar_ctx}"
        return base

    def get_prompt_content(self, ticker: str, market_data: dict) -> str:
        return """Analisis saham {ticker} sudah selesai. Berikut adalah laporan dari 4 analis:

1. Analis Teknikal:
{technical_report}

2. Analis Bullish:
{bull_report}

3. Analis Bearish:
{bear_report}

4. Analis Risiko:
{risk_report}


5. XGBoost ML Signal:
{ml_context}

TUGASMU:
1. KEPUTUSAN FINAL berdasarkan semua laporan di atas
2. Rekomendasi harga entry, stop loss, take profit, dan waktu hold
3. Executive summary dalam Bahasa Indonesia
4. Sertakan ml_anchor: bagaimana ML signal mempengaruhi keputusan

OUTPUT WAJIB JSON SAJA (tanpa markdown):
{{\"action\": \"STRONG_BUY\"|\"BUY\"|\"HOLD\"|\"SELL\"|\"STRONG_SELL\", \"stance\": \"BULLISH\"|\"BEARISH\"|\"NEUTRAL\", \"confidence\": 0.0-1.0, \"entry_price\": 1000.0, \"stop_loss\": 950.0, \"take_profit\": 1200.0, \"hold_days\": 30, \"ml_anchor\": \"pengaruh ML signal\", \"executive_summary\": \"150 kata Bahasa Indonesia\"}}""".format(
            ticker=ticker,
            technical_report=market_data.get("technical", {}),
            bull_report=market_data.get("bull", {}),
            bear_report=market_data.get("bear", {}),
            risk_report=market_data.get("risk", {}),
            ml_context=market_data.get("ml_context", "XGBoost signal tidak tersedia.")
        )
