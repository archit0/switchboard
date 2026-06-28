"""OpenAI-compatible HTTP server in front of the router.

Point any OpenAI client at it and it Just Works::

    client = OpenAI(base_url="http://localhost:8000/v1", api_key="anything")
    client.chat.completions.create(model="router", messages=[...])

Virtual models:
    router / router-balanced   triage -> single (easy) or Mixture-of-Agents (hard)
    router-cost                FrugalGPT cascade (cheapest; escalate on low score)
    router-quality             bias one tier up (best quality under Opus cost)

Any *real* model id (e.g. "claude-opus-4-8", "gemini-3-pro-preview") is passed
straight through to the gateway, so this also works as a plain proxy.
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from switchboard.engine import Engine

MODE_BY_MODEL = {
    "router": "balanced",
    "router-balanced": "balanced",
    "router-cost": "cost",
    "router-quality": "quality",
}

# Key under which router telemetry is attached to each response (ignored by
# standard OpenAI clients).
META_KEY = "switchboard"


@asynccontextmanager
async def _lifespan(app: FastAPI):
    app.state.engine = Engine()
    try:
        yield
    finally:
        await app.state.engine.aclose()


def create_app() -> FastAPI:
    app = FastAPI(title="switchboard", version="0.1.0", lifespan=_lifespan)

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"ok": True}

    @app.get("/v1/models")
    async def models() -> JSONResponse:
        virtual = [{"id": m, "object": "model", "owned_by": "switchboard"} for m in MODE_BY_MODEL]
        return JSONResponse({"object": "list", "data": virtual})

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        engine: Engine = request.app.state.engine
        body = await request.json()
        model = body.get("model", "router")
        messages = body.get("messages", [])
        stream = bool(body.get("stream", False))
        max_tokens = int(body.get("max_completion_tokens") or body.get("max_tokens") or 1024)

        # Real model id -> transparent proxy.
        if model not in MODE_BY_MODEL:
            comp = await engine.gw.complete(model, messages, max_tokens=max_tokens, temperature=body.get("temperature"))
            if not comp.ok and comp.error:
                return JSONResponse({"error": {"message": comp.error}}, status_code=502)
            shim = _ProxyResult(comp, model)
            if stream:
                return StreamingResponse(_sse_chunks(shim, model), media_type="text/event-stream")
            return JSONResponse(_openai_response(shim, model))

        rr = await engine.answer(messages, mode=MODE_BY_MODEL[model], max_tokens=max_tokens)
        if stream:
            return StreamingResponse(_sse_chunks(rr, model), media_type="text/event-stream")
        return JSONResponse(_openai_response(rr, model))

    return app


class _ProxyResult:
    """Adapts a raw Completion to the RouteResult-ish shape the formatter expects."""

    def __init__(self, comp, model: str):
        self.content = comp.content
        self.route = f"proxy:{model}"
        self.tier = "n/a"
        self.models_used = [model]
        self.prompt_tokens = comp.prompt_tokens
        self.completion_tokens = comp.completion_tokens
        self.cost = comp.cost
        self.baseline_cost_est = comp.cost
        self.savings_pct = 0.0
        self.latency_ms = comp.latency_ms
        self.cached = False
        self.steps: list = []


def _openai_response(rr, model_label: str) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_label,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": rr.content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": rr.prompt_tokens,
            "completion_tokens": rr.completion_tokens,
            "total_tokens": rr.prompt_tokens + rr.completion_tokens,
        },
        META_KEY: {
            "route": rr.route,
            "tier": rr.tier,
            "models_used": rr.models_used,
            "cost_usd": round(rr.cost, 8),
            "baseline_opus_cost_usd": round(rr.baseline_cost_est, 8),
            "savings_pct": round(rr.savings_pct, 1),
            "latency_ms": round(rr.latency_ms, 1),
            "cached": rr.cached,
            "steps": rr.steps,
        },
    }


def _sse_chunks(rr, model_label: str):
    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    def frame(delta: dict, finish=None) -> str:
        return (
            "data: "
            + json.dumps(
                {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_label,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                }
            )
            + "\n\n"
        )

    yield frame({"role": "assistant"})
    # We compute the full answer first (MoA can't truly stream), then chunk it
    # so streaming clients still work.
    text = rr.content
    step = max(1, len(text) // 40)
    for i in range(0, len(text), step):
        yield frame({"content": text[i : i + step]})
    yield frame({}, finish="stop")
    yield (
        "data: "
        + json.dumps(
            {
                META_KEY: {
                    "route": rr.route,
                    "cost_usd": round(rr.cost, 8),
                    "baseline_opus_cost_usd": round(rr.baseline_cost_est, 8),
                    "savings_pct": round(rr.savings_pct, 1),
                }
            }
        )
        + "\n\n"
    )
    yield "data: [DONE]\n\n"


# Module-level app for `uvicorn switchboard.server:app`.
app = create_app()


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)
