import asyncio
import httpx
import json
import re
import time
from typing import Dict, Any

class BaseAgent:
    def __init__(self, name: str, role: str, model: str, temperature: float, max_tokens: int, gateway_url: str, api_key: str):
        self.name = name
        self.role = role
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = httpx.AsyncClient(base_url=gateway_url, timeout=120.0) # 120 sec timeout
        self.api_key = api_key

    async def analyze(self, ticker: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.monotonic()
        prompt_content = self.get_prompt_content(ticker, market_data)

        messages = [
            {"role": "system", "content": self.get_system_prompt(ticker, market_data)},
            {"role": "user", "content": prompt_content},
        ]

        try:
            response = await self.client.post(
                "/chat/completions",
                headers={"Authorization": "Bearer {}".format(self.api_key)},
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                },
            )
            response.raise_for_status()

            # 9router returns SSE (text/event-stream) with "data: [DONE]" appended.
            # Different models differ: groq adds \n before data:[DONE], glm does not.
            # Solution: use raw_decode to parse ONLY the first JSON object from the body.
            raw_text = response.text
            idx = raw_text.find('{')
            if idx == -1:
                raise ValueError("No JSON in gateway response: " + raw_text[:300])
            raw_response, _ = json.JSONDecoder().raw_decode(raw_text, idx)

            # Extract message content from OpenAI-compatible response
            content = raw_response['choices'][0]['message']['content']

            # Robust JSON extraction from LLM content (may be wrapped in markdown ```json ```)
            cidx = content.find('{')
            if cidx == -1:
                raise ValueError("No JSON object in LLM content: " + content[:200])
            # raw_decode(content[cidx:]) starts at '{' position 0 of sliced string
            parsed_content, _ = json.JSONDecoder().raw_decode(content[cidx:])

            parsed_content['agent_name'] = self.name
            parsed_content['elapsed_s'] = time.monotonic() - start_time
            return parsed_content

        except (httpx.RequestError, httpx.HTTPStatusError, json.JSONDecodeError, KeyError, ValueError) as e:
            print("Agent {} failed to analyze {}: {}".format(self.name, ticker, repr(e)))
            return {
                "agent_name": self.name,
                "stance": "NEUTRAL",
                "confidence": 0.5,
                "key_points": ["Analysis failed due to error or timeout."],
                "reasoning": "Agent experienced an error or timeout.",
                "elapsed_s": time.monotonic() - start_time,
            }

    def get_system_prompt(self, ticker: str, market_data: Dict[str, Any]) -> str:
        raise NotImplementedError

    def get_prompt_content(self, ticker: str, market_data: Dict[str, Any]) -> str:
        return "Analisis saham {} dengan data berikut: {}".format(ticker, json.dumps(market_data, indent=2))