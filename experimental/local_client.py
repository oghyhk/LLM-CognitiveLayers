import json
import time
import logging
from openai import OpenAI

logger = logging.getLogger("daoti.local")


class LocalModelClient:
    def __init__(self, base_url: str = "http://localhost:11434/v1",
                 model: str = "gemma:2b", max_tokens: int = 512,
                 temperature: float = 0.7, timeout: int = 120):
        self.client = OpenAI(base_url=base_url, api_key="ollama", timeout=timeout)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = 2
        self.retry_delay = 1

    def chat(self, messages: list, system_prompt: str = None) -> str:
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
                logger.warning(f"Local API call attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise

    def chat_structured(self, messages: list, output_schema: dict,
                        task_instruction: str) -> dict:
        response_content, usage = self.chat(messages, task_instruction)
        result = self._try_parse_json(response_content)
        result["_usage"] = usage
        return result

    def _try_parse_json(self, text: str) -> dict:
        text = text.strip() if text else ""
        if not text:
            return {"response": "", "reasoning": "", "intent": ""}
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.startswith("```")]
            text = "\n".join(lines).strip()
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            pass
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start >= 0 and brace_end > brace_start:
            try:
                return json.loads(text[brace_start:brace_end + 1])
            except (json.JSONDecodeError, ValueError):
                pass
        return {"response": text, "reasoning": "", "intent": "", "_raw": True}
