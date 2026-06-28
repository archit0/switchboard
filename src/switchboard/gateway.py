"""Async client for the OpenAI-compatible gateway.

One client addresses every model — OpenAI, Anthropic and Google — by just
swapping the `model` field. This is what makes the router simple: it is just a
policy on top of a single `complete()` call.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import httpx

from . import config


@dataclass
class Completion:
    model: str
    content: str
    prompt_tokens: int
    completion_tokens: int
    cost: float
    latency_ms: float
    finish_reason: str = ""
    error: str | None = None
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.content.strip())


# gpt-5* reasoning models bill hidden reasoning tokens against the completion
# budget and return empty content if the budget is too small. We give them
# headroom and retry once with a bigger budget if they come back empty.
def _is_reasoning_model(model: str) -> bool:
    return model.startswith("gpt-5") or model.startswith("o3") or model.startswith("o4")


class Gateway:
    def __init__(self, cfg: config.GatewayConfig | None = None, *, concurrency: int = 24):
        self.cfg = cfg or config.GatewayConfig.from_env()
        self._client = httpx.AsyncClient(
            base_url=self.cfg.base_url,
            headers={
                "Authorization": f"Bearer {self.cfg.api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(120.0, connect=10.0),
            limits=httpx.Limits(max_connections=concurrency),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def complete(
        self,
        model: str,
        messages: list[dict],
        *,
        max_tokens: int = 1024,
        temperature: float | None = None,
        retries: int = 2,
    ) -> Completion:
        # gpt-5 reasoning models need extra headroom; everyone gets at least the ask.
        budget = max_tokens
        if _is_reasoning_model(model):
            budget = max(max_tokens, 1024) + 1024  # room for hidden reasoning

        last_err = "unknown error"
        for attempt in range(retries + 1):
            payload: dict = {
                "model": model,
                "messages": messages,
                "max_completion_tokens": budget,
            }
            # gpt-5 reasoning models reject non-default temperature.
            if temperature is not None and not _is_reasoning_model(model):
                payload["temperature"] = temperature

            t0 = time.perf_counter()
            try:
                r = await self._client.post("/chat/completions", json=payload)
                dt = (time.perf_counter() - t0) * 1000
                if r.status_code != 200:
                    last_err = f"HTTP {r.status_code}: {r.text[:200]}"
                    # 4xx other than rate limit won't fix on retry
                    if r.status_code not in (408, 409, 425, 429, 500, 502, 503, 504):
                        return Completion(model, "", 0, 0, 0.0, dt, error=last_err)
                    continue
                data = r.json()
            except Exception as e:  # noqa: BLE001
                dt = (time.perf_counter() - t0) * 1000
                last_err = f"{type(e).__name__}: {e}"
                continue

            choice = (data.get("choices") or [{}])[0]
            content = (choice.get("message") or {}).get("content") or ""
            finish = choice.get("finish_reason") or ""
            usage = data.get("usage") or {}
            pt = int(usage.get("prompt_tokens", 0))
            ct = int(usage.get("completion_tokens", 0))
            cost = config.cost_usd(model, pt, ct)

            # Empty content because the reasoning model ran out of budget — retry bigger.
            if not content.strip() and finish == "length" and attempt < retries:
                budget *= 2
                last_err = "empty content (finish_reason=length)"
                continue

            return Completion(
                model=model,
                content=content,
                prompt_tokens=pt,
                completion_tokens=ct,
                cost=cost,
                latency_ms=dt,
                finish_reason=finish,
                raw=data,
            )

        return Completion(model, "", 0, 0, 0.0, 0.0, error=last_err)
