from .base_agent import BaseAgent
import json

class OrchestratorAgent(BaseAgent):
    def get_system_prompt(self, ticker: str, market_data: dict) -> str:
        return """Kamu adalah Chief Investment Officer. Baca laporan 4 analis berikut dan buat KEPUTUSAN FINAL.
Bobot: Technical 30%, Bull 25%, Bear 25%, Risk 20%.
OUTPUT WAJIB JSON SAJA (tanpa markdown):
{{\"action\": \"STRONG_BUY\"|\"BUY\"|\"HOLD\"|\"SELL\"|\"STRONG_SELL\", \"confidence\": 0.0-1.0, \"entry_price\": 1000.0, \"stop_loss\": 950.0, \"take_profit\": 1200.0, \"hold_days\": 30, \"executive_summary\": \"150 kata Bahasa Indonesia\"}}"""

    def get_prompt_content(self, ticker: str, market_data: dict) -> str:
        # This agent will receive the reports from other agents
        # The actual content will be provided by the DebateEngine
        return """Analisis saham {ticker} sudah selesai. Berikut adalah laporan dari 4 analis:

1. Analis Teknikal:
{technical_report}

2. Analis Bullish:
{bull_report}

3. Analis Bearish:
{bear_report}

4. Analis Risiko:
{risk_report}


TUGASMU:
1. Berikan KEPUTUSAN FINAL berdasarkan laporan-laporan di atas
2. Berikan rekomendasi harga entry, stop loss, take profit, dan waktu hold
3. Berikan executive summary dalam Bahasa Indonesia

OUTPUT WAJIB JSON SAJA (tanpa markdown):
{{\"action\": \"STRONG_BUY\"|\"BUY\"|\"HOLD\"|\"SELL\"|\"STRONG_SELL\", \"confidence\": 0.0-1.0, \"entry_price\": 1000.0, \"stop_loss\": 950.0, \"take_profit\": 1200.0, \"hold_days\": 30, \"executive_summary\": \"150 kata Bahasa Indonesia\"}}""".format(
            ticker=ticker,
            technical_report=market_data.get('technical', {}),
            bull_report=market_data.get('bull', {}),
            bear_report=market_data.get('bear', {}),
            risk_report=market_data.get('risk', {})
        )
