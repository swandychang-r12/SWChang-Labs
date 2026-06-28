"""
Phase 5B: Dual-gateway BaseAgent
- Primary: 9router (groq models) at http://172.17.0.1:20128/v1
- Fallback: Ollama at http://172.17.0.1:11434 (OpenAI-compat /v1/chat/completions)
- Ordinal confidence: VERY_HIGH/HIGH/MEDIUM/LOW/INSUFFICIENT_DATA
"""
import asyncio
import httpx
import json
import re
import time
from typing import Dict, Any

OLLAMA_BASE_URL = "http://172.17.0.1:11434"
OLLAMA_FALLBACK_MODEL = "qwen2.5:7b"

ORDINAL_CONFIDENCE = {
    "VERY_HIGH": 0.90,
    "HIGH": 0.75,
    "MEDIUM": 0.55,
    "LOW": 0.35,
    "INSUFFICIENT_DATA": 0.20,
}


def ordinal_to_float(val) -> float:
    """Convert ordinal confidence string to float. Accept both formats."""
    if isinstance(val, (float, int)):
        return float(val)
    if isinstance(val, str):
        upper = val.upper().replace(" ", "_")
        return ORDINAL_CONFIDENCE.get(upper, 0.5)
    return 0.5


class BaseAgent:
    def __init__(
        self,
        name: str,
        role: str,
        model: str,
        temperature: float,
        max_tokens: int,
        gateway_url: str,
        api_key: str,
    ):
        self.name = name
        self.role = role
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.gateway_url = gateway_url
        self.api_key = api_key
        # Primary client (9router)
        self._primary_client = httpx.AsyncClient(base_url=gateway_url, timeout=90.0)
        # Fallback client (Ollama, OpenAI-compat format)
        self._fallback_client = httpx.AsyncClient(base_url=OLLAMA_BASE_URL, timeout=120.0)

    async def _call_llm(self, messages: list, use_fallback: bool = False) -> str:
        """Call LLM. Returns raw content string."""
        if use_fallback:
            # Ollama OpenAI-compat endpoint
            resp = await self._fallback_client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer ollama"},
                json={
                    "model": OLLAMA_FALLBACK_MODEL,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "stream": False,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        else:
            # 9router (primary) — SSE response, parse with raw_decode
            resp = await self._primary_client.post(
                "/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                },
            )
            resp.raise_for_status()
            raw_text = resp.text
            idx = raw_text.find("{")
            if idx == -1:
                raise ValueError("No JSON in gateway response: " + raw_text[:300])
            raw_response, _ = json.JSONDecoder().raw_decode(raw_text, idx)
            return raw_response["choices"][0]["message"]["content"]

    async def analyze(self, ticker: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.monotonic()
        prompt_content = self.get_prompt_content(ticker, market_data)
        messages = [
            {"role": "system", "content": self.get_system_prompt(ticker, market_data)},
            {"role": "user", "content": prompt_content},
        ]

        content = None
        gateway_used = "9router"
        try:
            content = await self._call_llm(messages, use_fallback=False)
        except Exception as e_primary:
            print(f"Agent {self.name}: 9router failed ({type(e_primary).__name__}: {e_primary}). Trying Ollama fallback...")
            try:
                content = await self._call_llm(messages, use_fallback=True)
                gateway_used = "ollama_fallback"
            except Exception as e_fallback:
                print(f"Agent {self.name}: Ollama fallback also failed: {e_fallback}")
                return {
                    "agent_name": self.name,
                    "stance": "NEUTRAL",
                    "confidence": 0.20,
                    "confidence_ordinal": "INSUFFICIENT_DATA",
                    "key_points": ["Analysis failed — both gateways unavailable."],
                    "reasoning": f"Primary: {e_primary} | Fallback: {e_fallback}",
                    "elapsed_s": time.monotonic() - start_time,
                    "gateway": "error",
                }

        try:
            # Extract JSON from LLM content
            cidx = content.find("{")
            if cidx == -1:
                raise ValueError("No JSON object in LLM content: " + content[:200])
            parsed, _ = json.JSONDecoder().raw_decode(content[cidx:])

            # Normalize ordinal confidence
            conf_raw = parsed.get("confidence", "MEDIUM")
            if isinstance(conf_raw, str) and conf_raw.upper() in ORDINAL_CONFIDENCE:
                parsed["confidence_ordinal"] = conf_raw.upper()
                parsed["confidence"] = ORDINAL_CONFIDENCE[conf_raw.upper()]
            else:
                fval = ordinal_to_float(conf_raw)
                # Map float back to ordinal
                if fval >= 0.85:
                    ordinal = "VERY_HIGH"
                elif fval >= 0.65:
                    ordinal = "HIGH"
                elif fval >= 0.45:
                    ordinal = "MEDIUM"
                elif fval >= 0.25:
                    ordinal = "LOW"
                else:
                    ordinal = "INSUFFICIENT_DATA"
                parsed["confidence_ordinal"] = ordinal
                parsed["confidence"] = fval

            parsed["agent_name"] = self.name
            parsed["gateway"] = gateway_used
            parsed["elapsed_s"] = time.monotonic() - start_time
            return parsed

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Agent {self.name} JSON parse failed: {e} | content: {content[:200]}")
            return {
                "agent_name": self.name,
                "stance": "NEUTRAL",
                "confidence": 0.20,
                "confidence_ordinal": "INSUFFICIENT_DATA",
                "key_points": ["JSON parse failed."],
                "reasoning": str(e),
                "elapsed_s": time.monotonic() - start_time,
                "gateway": gateway_used,
            }

    def get_system_prompt(self, ticker: str, market_data: Dict[str, Any]) -> str:
        raise NotImplementedError

    def get_prompt_content(self, ticker: str, market_data: Dict[str, Any]) -> str:
        return f"Analisis saham {ticker} dengan data berikut:\n{json.dumps(market_data, indent=2)}"
