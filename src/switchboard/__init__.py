"""switchboard — a smart, OpenAI-compatible LLM router that saves cost without losing quality.

More than a proxy: it's a tiny *agent* — a triage → verify → escalate loop with a
controller model, a judge, and a Mixture-of-Agents. Point any OpenAI client at the
switchboard server and it routes each request to the cheapest model that can handle
it — easy prompts to a small model, hard ones to a parallel Mixture-of-Agents —
trading a little latency for large savings while holding (or beating) frontier-model
quality on a representative workload.

Quickstart (library)::

    import asyncio
    from switchboard import Engine

    async def main():
        eng = Engine()
        result = await eng.answer(
            [{"role": "user", "content": "What is 17 * 23?"}],
            mode="cost",
        )
        print(result.content, result.cost, result.savings_pct)
        await eng.aclose()

    asyncio.run(main())

Quickstart (OpenAI-compatible server)::

    $ switchboard serve            # http://localhost:8000/v1

    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="anything")
    client.chat.completions.create(model="router-cost", messages=[...])

The gateway is configured via the ``OPENAI_BASE_URL`` and ``OPENAI_API_KEY``
environment variables (any OpenAI-compatible endpoint that fronts multiple
providers — e.g. a LiteLLM proxy — works).
"""

from switchboard.cache import ResponseCache
from switchboard.classify import Triage, triage
from switchboard.config import GatewayConfig, cost_usd, price_of
from switchboard.engine import Engine, RouteResult
from switchboard.gateway import Completion, Gateway

__version__ = "0.2.0"

__all__ = [
    "Completion",
    "Engine",
    "Gateway",
    "GatewayConfig",
    "ResponseCache",
    "RouteResult",
    "Triage",
    "__version__",
    "cost_usd",
    "price_of",
    "triage",
]
