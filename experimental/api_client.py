import json
import time
import logging
from openai import OpenAI

logger = logging.getLogger("daoti.api")


class APIClient:
    def __init__(self, base_url: str, api_key: str, model: str,
                 max_tokens: int = 1024, temperature: float = 0.7, timeout: int = 60):
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = 3
        self.retry_delay = 2

    def chat(self, messages: list, system_prompt: str = None) -> str:
        content, _ = self.chat_with_usage(messages, system_prompt)
        return content

    def chat_with_usage(self, messages: list, system_prompt: str = None) -> tuple:
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=full_messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                }
                return response.choices[0].message.content, usage
            except Exception as e:
                logger.warning(f"API call attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise

    def chat_structured(self, messages: list, output_schema: dict,
                        task_instruction: str) -> dict:
        schema_str = json.dumps(output_schema, indent=2)
        system_prompt = (
            f"{task_instruction}\n\n"
            f"Output ONLY valid JSON matching this schema. "
            f"No other text:\n{schema_str}"
        )
        for attempt in range(self.max_retries):
            try:
                raw, usage = self.chat_with_usage(messages, system_prompt)
                result = self._parse_json(raw)
                result["_usage"] = usage
                return result
            except (json.JSONDecodeError, ValueError) as e:
                raw_preview = str(raw)[:200] if 'raw' in dir() else "(no response)"
                logger.warning(
                    f"JSON parse attempt {attempt + 1} failed: {e}. "
                    f"Raw response preview: {raw_preview}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise ValueError(
                        f"API returned non-JSON after {self.max_retries} attempts. "
                        f"Raw preview: {raw_preview}"
                    )

    def _parse_json(self, text: str) -> dict:
        text = text.strip() if text else ""
        if not text:
            raise ValueError("Empty response from API")
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.startswith("```")]
            text = "\n".join(lines)
            text = text.strip()
        if not text:
            raise ValueError("Empty JSON content after stripping markdown")
        return json.loads(text)
