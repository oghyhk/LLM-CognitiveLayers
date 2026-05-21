import time
import logging
from openai import OpenAI

logger = logging.getLogger("daoti.local")


class LocalModelClient:
    def __init__(self, base_url: str = "http://localhost:11434/v1",
                 model: str = "gemma:2b", max_tokens: int = 256,
                 temperature: float = 0.7, timeout: int = 120):
        self.client = OpenAI(base_url=base_url, api_key="ollama", timeout=timeout)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = 2
        self.retry_delay = 1

    def chat(self, messages: list, system_prompt: str = None):
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=full_messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                if response.usage:
                    usage = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    }
                return response.choices[0].message.content or "", usage
            except Exception as e:
                logger.warning(f"Local API attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise
