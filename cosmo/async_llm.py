import asyncio
import json
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    import aiohttp
except ImportError:  # pragma: no cover - exercised only in minimal environments.
    aiohttp = None


class LLMRequestError(RuntimeError):
    pass


@dataclass
class RateLimitState:
    concurrency: int
    min_concurrency: int = 1
    max_concurrency: int = 64
    success_streak: int = 0
    cooldown_until: float = 0.0


class DynamicRateLimiter:
    """Adaptive concurrency limiter for OpenAI-compatible high-throughput APIs."""

    def __init__(self, initial_concurrency: int = 16, min_concurrency: int = 1, max_concurrency: int = 64):
        initial = max(min_concurrency, min(max_concurrency, initial_concurrency))
        self.state = RateLimitState(initial, min_concurrency, max_concurrency)
        self._active = 0
        self._condition = asyncio.Condition()

    async def acquire(self):
        async with self._condition:
            while True:
                now = time.monotonic()
                if now < self.state.cooldown_until:
                    await asyncio.sleep(self.state.cooldown_until - now)
                if self._active < self.state.concurrency:
                    self._active += 1
                    return
                await self._condition.wait()

    async def release(self, ok: bool, rate_limited: bool = False):
        async with self._condition:
            self._active = max(0, self._active - 1)
            if rate_limited:
                self.state.concurrency = max(self.state.min_concurrency, self.state.concurrency // 2)
                self.state.success_streak = 0
                self.state.cooldown_until = time.monotonic() + 2.0
            elif ok:
                self.state.success_streak += 1
                if self.state.success_streak >= max(4, self.state.concurrency):
                    self.state.concurrency = min(self.state.max_concurrency, self.state.concurrency + 1)
                    self.state.success_streak = 0
            else:
                self.state.concurrency = max(self.state.min_concurrency, self.state.concurrency - 1)
                self.state.success_streak = 0
            self._condition.notify_all()


class AsyncOpenAIChatClient:
    def __init__(
        self,
        api_base: str,
        api_key: str,
        model: str = "Qwen/Qwen2.5-72B-Instruct",
        timeout: int = 120,
        max_retries: int = 5,
        initial_concurrency: int = 16,
        max_concurrency: int = 64,
    ):
        if aiohttp is None:
            raise RuntimeError("aiohttp is required for AsyncOpenAIChatClient. Install dependencies with `pip install -r requirements.txt`.")
        base = api_base.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        self.url = f"{base}/v1/chat/completions"
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.rate_limiter = DynamicRateLimiter(
            initial_concurrency=initial_concurrency,
            min_concurrency=1,
            max_concurrency=max_concurrency,
        )
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        response_format: Optional[Dict[str, str]] = None,
    ) -> str:
        if self._session is None:
            raise RuntimeError("AsyncOpenAIChatClient must be used as an async context manager.")

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        last_error = ""
        for attempt in range(self.max_retries):
            await self.rate_limiter.acquire()
            ok = False
            rate_limited = False
            try:
                async with self._session.post(self.url, headers=self.headers, json=payload) as resp:
                    text = await resp.text()
                    if resp.status == 200:
                        data = json.loads(text)
                        ok = True
                        return data["choices"][0]["message"]["content"]
                    rate_limited = resp.status in {408, 409, 425, 429, 500, 502, 503, 504}
                    last_error = f"HTTP {resp.status}: {text[:500]}"
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                rate_limited = True
                last_error = repr(exc)
            finally:
                await self.rate_limiter.release(ok=ok, rate_limited=rate_limited)

            sleep_s = min(30.0, (2**attempt) + random.random())
            await asyncio.sleep(sleep_s)

        raise LLMRequestError(last_error or "request failed")

    async def json_chat(self, messages: List[Dict[str, str]], max_tokens: int = 512) -> Dict[str, Any]:
        content = await self.chat(
            messages=messages,
            temperature=0.0,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = cleaned.removeprefix("json").strip()
        return json.loads(cleaned)
